import atexit
import datetime
import fnmatch
import os
import queue
import sys
import tempfile
import threading
import time
from typing import List, Optional
from urllib.parse import quote as url_quote
import wandb
from wandb.proto import wandb_internal_pb2  # type: ignore
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import context, datastore, handler, sender, tb_watcher
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import filesystem
from wandb.util import check_and_warn_old
class SyncThread(threading.Thread):

    def __init__(self, sync_list, project=None, entity=None, run_id=None, job_type=None, view=None, verbose=None, mark_synced=None, app_url=None, sync_tensorboard=None, log_path=None, append=None, skip_console=None):
        threading.Thread.__init__(self)
        wandb._set_internal_process(disable=True)
        self._sync_list = sync_list
        self._project = project
        self._entity = entity
        self._run_id = run_id
        self._job_type = job_type
        self._view = view
        self._verbose = verbose
        self._mark_synced = mark_synced
        self._app_url = app_url
        self._sync_tensorboard = sync_tensorboard
        self._log_path = log_path
        self._append = append
        self._skip_console = skip_console
        self._tmp_dir = tempfile.TemporaryDirectory()
        atexit.register(self._tmp_dir.cleanup)

    def _parse_pb(self, data, exit_pb=None):
        pb = wandb_internal_pb2.Record()
        pb.ParseFromString(data)
        record_type = pb.WhichOneof('record_type')
        if self._view:
            if self._verbose:
                print('Record:', pb)
            else:
                print('Record:', record_type)
            return (pb, exit_pb, True)
        if record_type == 'run':
            if self._run_id:
                pb.run.run_id = self._run_id
            if self._project:
                pb.run.project = self._project
            if self._entity:
                pb.run.entity = self._entity
            if self._job_type:
                pb.run.job_type = self._job_type
            pb.control.req_resp = True
        elif record_type in ('output', 'output_raw') and self._skip_console:
            return (pb, exit_pb, True)
        elif record_type == 'exit':
            exit_pb = pb
            return (pb, exit_pb, True)
        elif record_type == 'final':
            assert exit_pb, 'final seen without exit'
            pb = exit_pb
            exit_pb = None
        return (pb, exit_pb, False)

    def _find_tfevent_files(self, sync_item):
        tb_event_files = 0
        tb_logdirs = []
        tb_root = None
        if self._sync_tensorboard:
            if os.path.isdir(sync_item):
                files = []
                for dirpath, _, _files in os.walk(sync_item):
                    for f in _files:
                        if TFEVENT_SUBSTRING in f:
                            files.append(os.path.join(dirpath, f))
                for tfevent in files:
                    tb_event_files += 1
                    tb_dir = os.path.dirname(os.path.abspath(tfevent))
                    if tb_dir not in tb_logdirs:
                        tb_logdirs.append(tb_dir)
                if len(tb_logdirs) > 0:
                    tb_root = os.path.dirname(os.path.commonprefix(tb_logdirs))
            elif TFEVENT_SUBSTRING in sync_item:
                tb_root = os.path.dirname(os.path.abspath(sync_item))
                tb_logdirs.append(tb_root)
                tb_event_files = 1
        return (tb_event_files, tb_logdirs, tb_root)

    def _setup_tensorboard(self, tb_root, tb_logdirs, tb_event_files, sync_item):
        """Return true if this sync item can be synced as tensorboard."""
        if tb_root is not None:
            if tb_event_files > 0 and sync_item.endswith(WANDB_SUFFIX):
                wandb.termwarn('Found .wandb file, not streaming tensorboard metrics.')
            else:
                print(f'Found {tb_event_files} tfevent files in {tb_root}')
                if len(tb_logdirs) > 3:
                    wandb.termwarn(f'Found {len(tb_logdirs)} directories containing tfevent files. If these represent multiple experiments, sync them individually or pass a list of paths.')
                return True
        return False

    def _send_tensorboard(self, tb_root, tb_logdirs, send_manager):
        if self._entity is None:
            viewer, _ = send_manager._api.viewer_server_info()
            self._entity = viewer.get('entity')
        proto_run = wandb_internal_pb2.RunRecord()
        proto_run.run_id = self._run_id or wandb.util.generate_id()
        proto_run.project = self._project or wandb.util.auto_project_name(None)
        proto_run.entity = self._entity
        proto_run.telemetry.feature.sync_tfevents = True
        url = '{}/{}/{}/runs/{}'.format(self._app_url, url_quote(proto_run.entity), url_quote(proto_run.project), url_quote(proto_run.run_id))
        print('Syncing: %s ...' % url)
        sys.stdout.flush()
        record_q = queue.Queue()
        sender_record_q = queue.Queue()
        new_interface = InterfaceQueue(record_q)
        context_keeper = context.ContextKeeper()
        send_manager = sender.SendManager(settings=send_manager._settings, record_q=sender_record_q, result_q=queue.Queue(), interface=new_interface, context_keeper=context_keeper)
        record = send_manager._interface._make_record(run=proto_run)
        settings = wandb.Settings(root_dir=self._tmp_dir.name, run_id=proto_run.run_id, _start_datetime=datetime.datetime.now(), _start_time=time.time())
        settings_static = SettingsStatic(settings.to_proto())
        handle_manager = handler.HandleManager(settings=settings_static, record_q=record_q, result_q=None, stopped=False, writer_q=sender_record_q, interface=new_interface, context_keeper=context_keeper)
        filesystem.mkdir_exists_ok(settings.files_dir)
        send_manager.send_run(record, file_dir=settings.files_dir)
        watcher = tb_watcher.TBWatcher(settings, proto_run, new_interface, True)
        for tb in tb_logdirs:
            watcher.add(tb, True, tb_root)
            sys.stdout.flush()
        watcher.finish()
        progress_step = 0
        spinner_states = ['-', '\\', '|', '/']
        line = ' Uploading data to wandb\r'
        while len(handle_manager) > 0:
            data = next(handle_manager)
            handle_manager.handle(data)
            while len(send_manager) > 0:
                data = next(send_manager)
                send_manager.send(data)
            print_line = spinner_states[progress_step % 4] + line
            wandb.termlog(print_line, newline=False, prefix=True)
            progress_step += 1
        while len(send_manager) > 0:
            data = next(send_manager)
            send_manager.send(data)
        sys.stdout.flush()
        handle_manager.finish()
        send_manager.finish()

    def _robust_scan(self, ds):
        """Attempt to scan data, handling incomplete files."""
        try:
            return ds.scan_data()
        except AssertionError as e:
            if ds.in_last_block():
                wandb.termwarn(f".wandb file is incomplete ({e}), be sure to sync this run again once it's finished")
                return None
            else:
                raise e

    def run(self):
        if self._log_path is not None:
            print(f'Find logs at: {self._log_path}')
        for sync_item in self._sync_list:
            tb_event_files, tb_logdirs, tb_root = self._find_tfevent_files(sync_item)
            if os.path.isdir(sync_item):
                files = os.listdir(sync_item)
                filtered_files = list(filter(lambda f: f.endswith(WANDB_SUFFIX), files))
                if tb_root is None and (check_and_warn_old(files) or len(filtered_files) != 1):
                    print(f'Skipping directory: {sync_item}')
                    continue
                if len(filtered_files) > 0:
                    sync_item = os.path.join(sync_item, filtered_files[0])
            sync_tb = self._setup_tensorboard(tb_root, tb_logdirs, tb_event_files, sync_item)
            root_dir = self._tmp_dir.name if sync_tb else os.path.dirname(sync_item)
            resume = 'allow' if self._append else None
            sm = sender.SendManager.setup(root_dir, resume=resume)
            if sync_tb:
                self._send_tensorboard(tb_root, tb_logdirs, sm)
                continue
            ds = datastore.DataStore()
            try:
                ds.open_for_scan(sync_item)
            except AssertionError as e:
                print(f'.wandb file is empty ({e}), skipping: {sync_item}')
                continue
            exit_pb = None
            finished = False
            shown = False
            while True:
                data = self._robust_scan(ds)
                if data is None:
                    break
                pb, exit_pb, cont = self._parse_pb(data, exit_pb)
                if exit_pb is not None:
                    finished = True
                if cont:
                    continue
                sm.send(pb)
                while not sm._record_q.empty():
                    data = sm._record_q.get(block=True)
                    sm.send(data)
                if pb.control.req_resp:
                    result = sm._result_q.get(block=True)
                    result_type = result.WhichOneof('result_type')
                    if not shown and result_type == 'run_result':
                        r = result.run_result.run
                        url = '{}/{}/{}/runs/{}'.format(self._app_url, url_quote(r.entity), url_quote(r.project), url_quote(r.run_id))
                        print('Syncing: %s ... ' % url, end='')
                        sys.stdout.flush()
                        shown = True
            sm.finish()
            if self._mark_synced and (not self._view) and finished:
                synced_file = f'{sync_item}{SYNCED_SUFFIX}'
                with open(synced_file, 'w'):
                    pass
            print('done.')