import itertools
import json
import logging
import numbers
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import patch
import filelock
import polars as pl
import requests
import urllib3
import yaml
from wandb_gql import gql
import wandb
import wandb.apis.reports as wr
from wandb.apis.public import ArtifactCollection, Run
from wandb.apis.public.files import File
from wandb.apis.reports import Report
from wandb.util import coalesce, remove_keys_with_none_values
from . import validation
from .internals import internal
from .internals.protocols import PathStr, Policy
from .internals.util import Namespace, for_each
class WandbImporter:
    """Transfers runs, reports, and artifact sequences between W&B instances."""

    def __init__(self, src_base_url: str, src_api_key: str, dst_base_url: str, dst_api_key: str, *, custom_api_kwargs: Optional[Dict[str, Any]]=None) -> None:
        self.src_base_url = src_base_url
        self.src_api_key = src_api_key
        self.dst_base_url = dst_base_url
        self.dst_api_key = dst_api_key
        if custom_api_kwargs is None:
            custom_api_kwargs = {'timeout': 600}
        self.src_api = wandb.Api(api_key=src_api_key, overrides={'base_url': src_base_url}, **custom_api_kwargs)
        self.dst_api = wandb.Api(api_key=dst_api_key, overrides={'base_url': dst_base_url}, **custom_api_kwargs)
        self.run_api_kwargs = {'src_base_url': src_base_url, 'src_api_key': src_api_key, 'dst_base_url': dst_base_url, 'dst_api_key': dst_api_key}

    def __repr__(self):
        return f'<WandbImporter src={self.src_base_url}, dst={self.dst_base_url}>'

    def _import_run(self, run: WandbRun, *, namespace: Optional[Namespace]=None, config: Optional[internal.SendManagerConfig]=None) -> None:
        """Import one WandbRun.

        Use `namespace` to specify alternate settings like where the run should be uploaded
        """
        if namespace is None:
            namespace = Namespace(run.entity(), run.project())
        if config is None:
            config = internal.SendManagerConfig(metadata=True, files=True, media=True, code=True, history=True, summary=True, terminal_output=True)
        settings_override = {'api_key': self.dst_api_key, 'base_url': self.dst_base_url, 'resume': 'true', 'resumed': True}
        logger.debug(f'Importing run, run={run!r}')
        internal.send_run(run, overrides=namespace.send_manager_overrides, settings_override=settings_override, config=config)
        if config.history:
            logger.debug(f'Collecting history artifacts, run={run!r}')
            history_arts = []
            for art in run.run.logged_artifacts():
                if art.type != 'wandb-history':
                    continue
                logger.debug(f'Collecting history artifact art.name={art.name!r}')
                new_art = _clone_art(art)
                history_arts.append(new_art)
            logger.debug(f'Importing history artifacts, run={run!r}')
            internal.send_run(run, extra_arts=history_arts, overrides=namespace.send_manager_overrides, settings_override=settings_override, config=config)

    def _delete_collection_in_dst(self, seq: ArtifactSequence, namespace: Optional[Namespace]=None):
        """Deletes the equivalent artifact collection in destination.

        Intended to clear the destination when an uploaded artifact does not pass validation.
        """
        entity = coalesce(namespace.entity, seq.entity)
        project = coalesce(namespace.project, seq.project)
        art_type = f'{entity}/{project}/{seq.type_}'
        art_name = seq.name
        logger.info(f'Deleting collection entity={entity!r}, project={project!r}, art_type={art_type!r}, art_name={art_name!r}')
        try:
            dst_collection = self.dst_api.artifact_collection(art_type, art_name)
        except (wandb.CommError, ValueError):
            logger.warn(f"Collection doesn't exist art_type={art_type!r}, art_name={art_name!r}")
            return
        try:
            dst_collection.delete()
        except (wandb.CommError, ValueError) as e:
            logger.warn(f"Collection can't be deleted, art_type={art_type!r}, art_name={art_name!r}, e={e!r}")
            return

    def _import_artifact_sequence(self, seq: ArtifactSequence, *, namespace: Optional[Namespace]=None) -> None:
        """Import one artifact sequence.

        Use `namespace` to specify alternate settings like where the artifact sequence should be uploaded
        """
        if not seq.artifacts:
            logger.warn(f'Artifact seq={seq!r} has no artifacts, skipping.')
            return
        if namespace is None:
            namespace = Namespace(seq.entity, seq.project)
        settings_override = {'api_key': self.dst_api_key, 'base_url': self.dst_base_url, 'resume': 'true', 'resumed': True}
        send_manager_config = internal.SendManagerConfig(log_artifacts=True)
        self._delete_collection_in_dst(seq, namespace)
        art = seq.artifacts[0]
        run_or_dummy: Optional[Run] = _get_run_or_dummy_from_art(art, self.src_api)
        groups_of_artifacts = list(_make_groups_of_artifacts(seq))
        for i, group in enumerate(groups_of_artifacts, 1):
            art = group[0]
            if art.description == ART_SEQUENCE_DUMMY_PLACEHOLDER:
                run = WandbRun(run_or_dummy, **self.run_api_kwargs)
            else:
                try:
                    wandb_run = art.logged_by()
                except ValueError:
                    pass
                if wandb_run is None:
                    logger.warn(f'Run for art.name={art.name!r} does not exist (deleted?), using run_or_dummy={run_or_dummy!r}')
                    wandb_run = run_or_dummy
                new_art = _clone_art(art)
                group = [new_art]
                run = WandbRun(wandb_run, **self.run_api_kwargs)
            logger.info(f'Uploading partial artifact seq={seq!r}, {i}/{len(groups_of_artifacts)}')
            internal.send_run(run, extra_arts=group, overrides=namespace.send_manager_overrides, settings_override=settings_override, config=send_manager_config)
        logger.info(f'Finished uploading seq={seq!r}')
        self._remove_placeholders(seq)

    def _remove_placeholders(self, seq: ArtifactSequence) -> None:
        try:
            retry_arts_func = internal.exp_retry(self._dst_api.artifacts)
            dst_arts = list(retry_arts_func(seq.type_, seq.name))
        except wandb.CommError:
            logger.warn(f'seq={seq!r} does not exist in dst.  Has it already been deleted?')
            return
        except TypeError as e:
            logger.error(f'Problem getting dst versions (try again later) e={e!r}')
            return
        for art in dst_arts:
            if art.description != ART_SEQUENCE_DUMMY_PLACEHOLDER:
                continue
            if art.type in ('wandb-history', 'job'):
                continue
            try:
                art.delete(delete_aliases=True)
            except wandb.CommError as e:
                if 'cannot delete system managed artifact' in str(e):
                    logger.warn('Cannot delete system managed artifact')
                else:
                    raise e

    def _get_dst_art(self, src_art: Run, entity: Optional[str]=None, project: Optional[str]=None) -> Artifact:
        entity = coalesce(entity, src_art.entity)
        project = coalesce(project, src_art.project)
        name = src_art.name
        return self.dst_api.artifact(f'{entity}/{project}/{name}')

    def _get_run_problems(self, src_run: Run, dst_run: Run, force_retry: bool=False) -> List[dict]:
        problems = []
        if force_retry:
            problems.append('__force_retry__')
        if (non_matching_metadata := self._compare_run_metadata(src_run, dst_run)):
            problems.append('metadata:' + str(non_matching_metadata))
        if (non_matching_summary := self._compare_run_summary(src_run, dst_run)):
            problems.append('summary:' + str(non_matching_summary))
        return problems

    def _compare_run_metadata(self, src_run: Run, dst_run: Run) -> dict:
        fname = 'wandb-metadata.json'
        src_f = src_run.file(fname)
        if src_f.size == 0:
            return {}
        dst_f = dst_run.file(fname)
        try:
            contents = wandb.util.download_file_into_memory(dst_f.url, self.dst_api.api_key)
        except urllib3.exceptions.ReadTimeoutError:
            return {'Error checking': 'Timeout'}
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return {'Bad upload': f'File not found: {fname}'}
            return {'http problem': f'{fname}: ({e})'}
        dst_meta = wandb.wandb_sdk.lib.json_util.loads(contents)
        non_matching = {}
        if src_run.metadata:
            for k, src_v in src_run.metadata.items():
                if k not in dst_meta:
                    non_matching[k] = {'src': src_v, 'dst': 'KEY NOT FOUND'}
                    continue
                dst_v = dst_meta[k]
                if src_v != dst_v:
                    non_matching[k] = {'src': src_v, 'dst': dst_v}
        return non_matching

    def _compare_run_summary(self, src_run: Run, dst_run: Run) -> dict:
        non_matching = {}
        for k, src_v in src_run.summary.items():
            if isinstance(src_v, str) and src_v.startswith('wandb-client-artifact://'):
                continue
            if k in ('_wandb', '_runtime'):
                continue
            src_v = _recursive_cast_to_dict(src_v)
            dst_v = dst_run.summary.get(k)
            dst_v = _recursive_cast_to_dict(dst_v)
            if isinstance(src_v, dict) and isinstance(dst_v, dict):
                for kk, sv in src_v.items():
                    if isinstance(sv, str) and sv.startswith('wandb-client-artifact://'):
                        continue
                    dv = dst_v.get(kk)
                    if not _almost_equal(sv, dv):
                        non_matching[f'{k}-{kk}'] = {'src': sv, 'dst': dv}
            elif not _almost_equal(src_v, dst_v):
                non_matching[k] = {'src': src_v, 'dst': dst_v}
        return non_matching

    def _collect_failed_artifact_sequences(self) -> Iterable[ArtifactSequence]:
        if (df := _read_ndjson(ARTIFACT_ERRORS_FNAME)) is None:
            logger.debug(f'ARTIFACT_ERRORS_FNAME={ARTIFACT_ERRORS_FNAME!r} is empty, returning nothing')
            return
        unique_failed_sequences = df[['src_entity', 'src_project', 'name', 'type']].unique()
        for row in unique_failed_sequences.iter_rows(named=True):
            entity = row['src_entity']
            project = row['src_project']
            name = row['name']
            _type = row['type']
            art_name = f'{entity}/{project}/{name}'
            arts = self.src_api.artifacts(_type, art_name)
            arts = sorted(arts, key=lambda a: int(a.version.lstrip('v')))
            arts = sorted(arts, key=lambda a: a.type)
            yield ArtifactSequence(arts, entity, project, _type, name)

    def _cleanup_dummy_runs(self, *, namespaces: Optional[Iterable[Namespace]]=None, api: Optional[Api]=None, remapping: Optional[Dict[Namespace, Namespace]]=None) -> None:
        api = coalesce(api, self.dst_api)
        namespaces = coalesce(namespaces, self._all_namespaces())
        for ns in namespaces:
            if remapping and ns in remapping:
                ns = remapping[ns]
            logger.debug(f'Cleaning up, ns={ns!r}')
            try:
                runs = list(api.runs(ns.path, filters={'displayName': RUN_DUMMY_PLACEHOLDER}))
            except ValueError as e:
                if 'Could not find project' in str(e):
                    logger.error('Could not find project, does it exist?')
                    continue
            for run in runs:
                logger.debug(f'Deleting dummy run={run!r}')
                run.delete(delete_artifacts=False)

    def _import_report(self, report: Report, *, namespace: Optional[Namespace]=None) -> None:
        """Import one wandb.Report.

        Use `namespace` to specify alternate settings like where the report should be uploaded
        """
        if namespace is None:
            namespace = Namespace(report.entity, report.project)
        entity = coalesce(namespace.entity, report.entity)
        project = coalesce(namespace.project, report.project)
        name = report.name
        title = report.title
        description = report.description
        api = self.dst_api
        logger.debug(f'Upserting entity={entity!r}/project={project!r}')
        try:
            api.create_project(project, entity)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 409:
                logger.warn(f'Issue upserting entity={entity!r}/project={project!r}, e={e!r}')
        logger.debug(f'Upserting report entity={entity!r}, project={project!r}, name={name!r}, title={title!r}')
        api.client.execute(wr.report.UPSERT_VIEW, variable_values={'id': None, 'name': name, 'entityName': entity, 'projectName': project, 'description': description, 'displayName': title, 'type': 'runs', 'spec': json.dumps(report.spec)})

    def _use_artifact_sequence(self, sequence: ArtifactSequence, *, namespace: Optional[Namespace]=None):
        if namespace is None:
            namespace = Namespace(sequence.entity, sequence.project)
        settings_override = {'api_key': self.dst_api_key, 'base_url': self.dst_base_url, 'resume': 'true', 'resumed': True}
        logger.debug(f'Using artifact sequence with settings_override={settings_override!r}, namespace={namespace!r}')
        send_manager_config = internal.SendManagerConfig(use_artifacts=True)
        for art in sequence:
            if (used_by := art.used_by()) is None:
                continue
            for wandb_run in used_by:
                run = WandbRun(wandb_run, **self.run_api_kwargs)
                internal.send_run(run, overrides=namespace.send_manager_overrides, settings_override=settings_override, config=send_manager_config)

    def import_runs(self, *, namespaces: Optional[Iterable[Namespace]]=None, remapping: Optional[Dict[Namespace, Namespace]]=None, parallel: bool=True, incremental: bool=True, max_workers: Optional[int]=None, limit: Optional[int]=None, metadata: bool=True, files: bool=True, media: bool=True, code: bool=True, history: bool=True, summary: bool=True, terminal_output: bool=True):
        logger.info('START: Import runs')
        logger.info('Setting up for import')
        _create_files_if_not_exists()
        _clear_fname(RUN_ERRORS_FNAME)
        logger.info('Collecting runs')
        runs = list(self._collect_runs(namespaces=namespaces, limit=limit))
        logger.info(f'Validating runs, len(runs)={len(runs)!r}')
        self._validate_runs(runs, skip_previously_validated=incremental, remapping=remapping)
        logger.info('Collecting failed runs')
        runs = list(self._collect_failed_runs())
        logger.info(f'Importing runs, len(runs)={len(runs)!r}')

        def _import_run_wrapped(run):
            namespace = Namespace(run.entity(), run.project())
            if remapping is not None and namespace in remapping:
                namespace = remapping[namespace]
            config = internal.SendManagerConfig(metadata=metadata, files=files, media=media, code=code, history=history, summary=summary, terminal_output=terminal_output)
            logger.debug(f'Importing run={run!r}, namespace={namespace!r}, config={config!r}')
            self._import_run(run, namespace=namespace, config=config)
            logger.debug(f'Finished importing run={run!r}, namespace={namespace!r}, config={config!r}')
        for_each(_import_run_wrapped, runs, max_workers=max_workers, parallel=parallel)
        logger.info('END: Importing runs')

    def import_reports(self, *, namespaces: Optional[Iterable[Namespace]]=None, limit: Optional[int]=None, remapping: Optional[Dict[Namespace, Namespace]]=None):
        logger.info('START: Importing reports')
        logger.info('Collecting reports')
        reports = self._collect_reports(namespaces=namespaces, limit=limit)
        logger.info('Importing reports')

        def _import_report_wrapped(report):
            namespace = Namespace(report.entity, report.project)
            if remapping is not None and namespace in remapping:
                namespace = remapping[namespace]
            logger.debug(f'Importing report={report!r}, namespace={namespace!r}')
            self._import_report(report, namespace=namespace)
            logger.debug(f'Finished importing report={report!r}, namespace={namespace!r}')
        for_each(_import_report_wrapped, reports)
        logger.info('END: Importing reports')

    def import_artifact_sequences(self, *, namespaces: Optional[Iterable[Namespace]]=None, incremental: bool=True, max_workers: Optional[int]=None, remapping: Optional[Dict[Namespace, Namespace]]=None):
        """Import all artifact sequences from `namespaces`.

        Note: There is a known bug with the AWS backend where artifacts > 2048MB will fail to upload.  This seems to be related to multipart uploads, but we don't have a fix yet.
        """
        logger.info('START: Importing artifact sequences')
        _clear_fname(ARTIFACT_ERRORS_FNAME)
        logger.info('Collecting artifact sequences')
        seqs = list(self._collect_artifact_sequences(namespaces=namespaces))
        logger.info('Validating artifact sequences')
        self._validate_artifact_sequences(seqs, incremental=incremental, remapping=remapping)
        logger.info('Collecting failed artifact sequences')
        seqs = list(self._collect_failed_artifact_sequences())
        logger.info(f'Importing artifact sequences, len(seqs)={len(seqs)!r}')

        def _import_artifact_sequence_wrapped(seq):
            namespace = Namespace(seq.entity, seq.project)
            if remapping is not None and namespace in remapping:
                namespace = remapping[namespace]
            logger.debug(f'Importing artifact sequence seq={seq!r}, namespace={namespace!r}')
            self._import_artifact_sequence(seq, namespace=namespace)
            logger.debug(f'Finished importing artifact sequence seq={seq!r}, namespace={namespace!r}')
        for_each(_import_artifact_sequence_wrapped, seqs, max_workers=max_workers)
        logger.debug(f'Using artifact sequences, len(seqs)={len(seqs)!r}')

        def _use_artifact_sequence_wrapped(seq):
            namespace = Namespace(seq.entity, seq.project)
            if remapping is not None and namespace in remapping:
                namespace = remapping[namespace]
            logger.debug(f'Using artifact sequence seq={seq!r}, namespace={namespace!r}')
            self._use_artifact_sequence(seq, namespace=namespace)
            logger.debug(f'Finished using artifact sequence seq={seq!r}, namespace={namespace!r}')
        for_each(_use_artifact_sequence_wrapped, seqs, max_workers=max_workers)
        logger.info('Cleaning up dummy runs')
        self._cleanup_dummy_runs(namespaces=namespaces, remapping=remapping)
        logger.info('END: Importing artifact sequences')

    def import_all(self, *, runs: bool=True, artifacts: bool=True, reports: bool=True, namespaces: Optional[Iterable[Namespace]]=None, incremental: bool=True, remapping: Optional[Dict[Namespace, Namespace]]=None):
        logger.info(f'START: Importing all, runs={runs!r}, artifacts={artifacts!r}, reports={reports!r}')
        if runs:
            self.import_runs(namespaces=namespaces, incremental=incremental, remapping=remapping)
        if reports:
            self.import_reports(namespaces=namespaces, remapping=remapping)
        if artifacts:
            self.import_artifact_sequences(namespaces=namespaces, incremental=incremental, remapping=remapping)
        logger.info('END: Importing all')

    def _validate_run(self, src_run: Run, *, remapping: Optional[Dict[Namespace, Namespace]]=None) -> None:
        namespace = Namespace(src_run.entity, src_run.project)
        if remapping is not None and namespace in remapping:
            namespace = remapping[namespace]
        dst_entity = namespace.entity
        dst_project = namespace.project
        run_id = src_run.id
        try:
            dst_run = self.dst_api.run(f'{dst_entity}/{dst_project}/{run_id}')
        except wandb.CommError:
            problems = [f'run does not exist in dst at dst_entity={dst_entity!r}/dst_project={dst_project!r}']
        else:
            problems = self._get_run_problems(src_run, dst_run)
        d = {'src_entity': src_run.entity, 'src_project': src_run.project, 'dst_entity': dst_entity, 'dst_project': dst_project, 'run_id': run_id}
        if problems:
            d['problems'] = problems
            fname = RUN_ERRORS_FNAME
        else:
            fname = RUN_SUCCESSES_FNAME
        with filelock.FileLock('runs.lock'):
            with open(fname, 'a') as f:
                f.write(json.dumps(d) + '\n')

    def _filter_previously_checked_runs(self, runs: Iterable[Run], *, remapping: Optional[Dict[Namespace, Namespace]]=None) -> Iterable[Run]:
        if (df := _read_ndjson(RUN_SUCCESSES_FNAME)) is None:
            logger.debug(f'RUN_SUCCESSES_FNAME={RUN_SUCCESSES_FNAME!r} is empty, yielding all runs')
            yield from runs
            return
        data = []
        for r in runs:
            namespace = Namespace(r.entity, r.project)
            if remapping is not None and namespace in remapping:
                namespace = remapping[namespace]
            data.append({'src_entity': r.entity, 'src_project': r.project, 'dst_entity': namespace.entity, 'dst_project': namespace.project, 'run_id': r.id, 'data': r})
        df2 = pl.DataFrame(data)
        logger.debug(f'Starting with len(runs)={len(runs)!r} in namespaces')
        results = df2.join(df, how='anti', on=['src_entity', 'src_project', 'dst_entity', 'dst_project', 'run_id'])
        logger.debug(f'After filtering out already successful runs, len(results)={len(results)!r}')
        if not results.is_empty():
            results = results.filter(~results['run_id'].is_null())
            results = results.unique(['src_entity', 'src_project', 'dst_entity', 'dst_project', 'run_id'])
        for r in results.iter_rows(named=True):
            yield r['data']

    def _validate_artifact(self, src_art: Artifact, dst_entity: str, dst_project: str, download_files_and_compare: bool=False, check_entries_are_downloadable: bool=True):
        problems = []
        ignore_patterns = ['^job-(.*?)\\.py(:v\\d+)?$']
        for pattern in ignore_patterns:
            if re.search(pattern, src_art.name):
                return (src_art, dst_entity, dst_project, problems)
        try:
            dst_art = self._get_dst_art(src_art, dst_entity, dst_project)
        except Exception:
            problems.append('destination artifact not found')
            return (src_art, dst_entity, dst_project, problems)
        try:
            logger.debug('Comparing artifact manifests')
        except Exception as e:
            problems.append(f'Problem getting problems! problem with src_art.entity={src_art.entity!r}, src_art.project={src_art.project!r}, src_art.name={src_art.name!r} e={e!r}')
        else:
            problems += validation._compare_artifact_manifests(src_art, dst_art)
        if check_entries_are_downloadable:
            validation._check_entries_are_downloadable(dst_art)
        if download_files_and_compare:
            logger.debug(f'Downloading src_art={src_art!r}')
            try:
                src_dir = _download_art(src_art, root=f'{SRC_ART_PATH}/{src_art.name}')
            except requests.HTTPError as e:
                problems.append(f'Invalid download link for src src_art.entity={src_art.entity!r}, src_art.project={src_art.project!r}, src_art.name={src_art.name!r}, {e}')
            logger.debug(f'Downloading dst_art={dst_art!r}')
            try:
                dst_dir = _download_art(dst_art, root=f'{DST_ART_PATH}/{dst_art.name}')
            except requests.HTTPError as e:
                problems.append(f'Invalid download link for dst dst_art.entity={dst_art.entity!r}, dst_art.project={dst_art.project!r}, dst_art.name={dst_art.name!r}, {e}')
            else:
                logger.debug(f'Comparing artifact dirs src_dir={src_dir!r}, dst_dir={dst_dir!r}')
                if (problem := validation._compare_artifact_dirs(src_dir, dst_dir)):
                    problems.append(problem)
        return (src_art, dst_entity, dst_project, problems)

    def _validate_runs(self, runs: Iterable[WandbRun], *, skip_previously_validated: bool=True, remapping: Optional[Dict[Namespace, Namespace]]=None):
        base_runs = [r.run for r in runs]
        if skip_previously_validated:
            base_runs = list(self._filter_previously_checked_runs(base_runs, remapping=remapping))

        def _validate_run(run):
            logger.debug(f'Validating run={run!r}')
            self._validate_run(run, remapping=remapping)
            logger.debug(f'Finished validating run={run!r}')
        for_each(_validate_run, base_runs)

    def _collect_failed_runs(self):
        if (df := _read_ndjson(RUN_ERRORS_FNAME)) is None:
            logger.debug(f'RUN_ERRORS_FNAME={RUN_ERRORS_FNAME!r} is empty, returning nothing')
            return
        unique_failed_runs = df[['src_entity', 'src_project', 'dst_entity', 'dst_project', 'run_id']].unique()
        for row in unique_failed_runs.iter_rows(named=True):
            src_entity = row['src_entity']
            src_project = row['src_project']
            run_id = row['run_id']
            run = self.src_api.run(f'{src_entity}/{src_project}/{run_id}')
            yield WandbRun(run, **self.run_api_kwargs)

    def _filter_previously_checked_artifacts(self, seqs: Iterable[ArtifactSequence]):
        if (df := _read_ndjson(ARTIFACT_SUCCESSES_FNAME)) is None:
            logger.info(f'ARTIFACT_SUCCESSES_FNAME={ARTIFACT_SUCCESSES_FNAME!r} is empty, yielding all artifact sequences')
            for seq in seqs:
                yield from seq.artifacts
            return
        for seq in seqs:
            for art in seq:
                try:
                    logged_by = _get_run_or_dummy_from_art(art, self.src_api)
                except requests.HTTPError as e:
                    logger.error(f'Failed to get run, skipping: art={art!r}, e={e!r}')
                    continue
                if art.type == 'wandb-history' and isinstance(logged_by, _DummyRun):
                    logger.debug(f'Skipping history artifact art={art!r}')
                    continue
                entity = art.entity
                project = art.project
                _type = art.type
                name, ver = _get_art_name_ver(art)
                filtered_df = df.filter((df['src_entity'] == entity) & (df['src_project'] == project) & (df['name'] == name) & (df['version'] == ver) & (df['type'] == _type))
                if len(filtered_df) == 0:
                    yield art

    def _validate_artifact_sequences(self, seqs: Iterable[ArtifactSequence], *, incremental: bool=True, download_files_and_compare: bool=False, check_entries_are_downloadable: bool=True, remapping: Optional[Dict[Namespace, Namespace]]=None):
        if incremental:
            logger.info('Validating in incremental mode')

            def filtered_sequences():
                for seq in seqs:
                    if not seq.artifacts:
                        continue
                    art = seq.artifacts[0]
                    try:
                        logged_by = _get_run_or_dummy_from_art(art, self.src_api)
                    except requests.HTTPError as e:
                        logger.error(f'Validate Artifact http error: art.entity={art.entity!r}, art.project={art.project!r}, art.name={art.name!r}, e={e!r}')
                        continue
                    if art.type == 'wandb-history' and isinstance(logged_by, _DummyRun):
                        continue
                    yield seq
            artifacts = self._filter_previously_checked_artifacts(filtered_sequences())
        else:
            logger.info('Validating in non-incremental mode')
            artifacts = [art for seq in seqs for art in seq.artifacts]

        def _validate_artifact_wrapped(args):
            art, entity, project = args
            if remapping is not None and (namespace := Namespace(entity, project)) in remapping:
                remapped_ns = remapping[namespace]
                entity = remapped_ns.entity
                project = remapped_ns.project
            logger.debug(f'Validating art={art!r}, entity={entity!r}, project={project!r}')
            result = self._validate_artifact(art, entity, project, download_files_and_compare=download_files_and_compare, check_entries_are_downloadable=check_entries_are_downloadable)
            logger.debug(f'Finished validating art={art!r}, entity={entity!r}, project={project!r}')
            return result
        args = ((art, art.entity, art.project) for art in artifacts)
        art_problems = for_each(_validate_artifact_wrapped, args)
        for art, dst_entity, dst_project, problems in art_problems:
            name, ver = _get_art_name_ver(art)
            d = {'src_entity': art.entity, 'src_project': art.project, 'dst_entity': dst_entity, 'dst_project': dst_project, 'name': name, 'version': ver, 'type': art.type}
            if problems:
                d['problems'] = problems
                fname = ARTIFACT_ERRORS_FNAME
            else:
                fname = ARTIFACT_SUCCESSES_FNAME
            with open(fname, 'a') as f:
                f.write(json.dumps(d) + '\n')

    def _collect_runs(self, *, namespaces: Optional[Iterable[Namespace]]=None, limit: Optional[int]=None, skip_ids: Optional[List[str]]=None, start_date: Optional[str]=None, api: Optional[Api]=None) -> Iterable[WandbRun]:
        api = coalesce(api, self.src_api)
        namespaces = coalesce(namespaces, self._all_namespaces())
        filters: Dict[str, Any] = {}
        if skip_ids is not None:
            filters['name'] = {'$nin': skip_ids}
        if start_date is not None:
            filters['createdAt'] = {'$gte': start_date}

        def _runs():
            for ns in namespaces:
                logger.debug(f'Collecting runs from ns={ns!r}')
                for run in api.runs(ns.path, filters=filters):
                    yield WandbRun(run, **self.run_api_kwargs)
        runs = itertools.islice(_runs(), limit)
        yield from runs

    def _all_namespaces(self, *, entity: Optional[str]=None, api: Optional[Api]=None):
        api = coalesce(api, self.src_api)
        entity = coalesce(entity, api.default_entity)
        projects = api.projects(entity)
        for p in projects:
            yield Namespace(p.entity, p.name)

    def _collect_reports(self, *, namespaces: Optional[Iterable[Namespace]]=None, limit: Optional[int]=None, api: Optional[Api]=None):
        api = coalesce(api, self.src_api)
        namespaces = coalesce(namespaces, self._all_namespaces())
        wandb.login(key=self.src_api_key, host=self.src_base_url)

        def reports():
            for ns in namespaces:
                for r in api.reports(ns.path):
                    yield wr.Report.from_url(r.url, api=api)
        yield from itertools.islice(reports(), limit)

    def _collect_artifact_sequences(self, *, namespaces: Optional[Iterable[Namespace]]=None, limit: Optional[int]=None, api: Optional[Api]=None):
        api = coalesce(api, self.src_api)
        namespaces = coalesce(namespaces, self._all_namespaces())

        def artifact_sequences():
            for ns in namespaces:
                logger.debug(f'Collecting artifact sequences from ns={ns!r}')
                types = []
                try:
                    types = [t for t in api.artifact_types(ns.path)]
                except Exception as e:
                    logger.error(f'Failed to get artifact types e={e!r}')
                for t in types:
                    collections = []
                    if t.name == 'wandb-history':
                        continue
                    try:
                        collections = t.collections()
                    except Exception as e:
                        logger.error(f'Failed to get artifact collections e={e!r}')
                    for c in collections:
                        if c.is_sequence():
                            yield ArtifactSequence.from_collection(c)
        seqs = itertools.islice(artifact_sequences(), limit)
        unique_sequences = {seq.identifier: seq for seq in seqs}
        yield from unique_sequences.values()