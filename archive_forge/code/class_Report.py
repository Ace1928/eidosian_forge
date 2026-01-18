import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
@dataclass(config=dataclass_config)
class Report(Base):
    project: str
    entity: str = Field(default_factory=lambda: _get_api().default_entity)
    title: str = Field('Untitled Report', max_length=128)
    width: ReportWidth = 'readable'
    description: str = ''
    blocks: LList[BlockTypes] = Field(default_factory=list)
    id: str = Field(default_factory=lambda: '', init=False, repr=False, kw_only=True)
    _discussion_threads: list = Field(default_factory=list, init=False, repr=False)
    _ref: dict = Field(default_factory=dict, init=False, repr=False)
    _panel_settings: dict = Field(default_factory=dict, init=False, repr=False)
    _authors: LList[dict] = Field(default_factory=list, init=False, repr=False)
    _created_at: Optional[datetime] = Field(default_factory=lambda: None, init=False, repr=False)
    _updated_at: Optional[datetime] = Field(default_factory=lambda: None, init=False, repr=False)

    def to_model(self):
        blocks = self.blocks
        if len(blocks) > 0 and blocks[0] != P():
            blocks = [P()] + blocks
        if len(blocks) > 0 and blocks[-1] != P():
            blocks = blocks + [P()]
        if not blocks:
            blocks = [P(), P()]
        return internal.ReportViewspec(display_name=self.title, description=self.description, project=internal.Project(name=self.project, entity_name=self.entity), id=self.id, created_at=self._created_at, updated_at=self._updated_at, spec=internal.Spec(panel_settings=self._panel_settings, blocks=[b.to_model() for b in blocks], width=self.width, authors=self._authors, discussion_threads=self._discussion_threads, ref=self._ref))

    @classmethod
    def from_model(cls, model: internal.ReportViewspec):
        blocks = model.spec.blocks
        if blocks[0] == internal.Paragraph():
            blocks = blocks[1:]
        if blocks[-1] == internal.Paragraph():
            blocks = blocks[:-1]
        obj = cls(title=model.display_name, description=model.description, entity=model.project.entity_name, project=model.project.name, blocks=[_lookup(b) for b in blocks])
        obj.id = model.id
        obj._discussion_threads = model.spec.discussion_threads
        obj._panel_settings = model.spec.panel_settings
        obj._ref = model.spec.ref
        obj._authors = model.spec.authors
        obj._created_at = model.created_at
        obj._updated_at = model.updated_at
        return obj

    @property
    def url(self):
        if self.id == '':
            raise AttributeError('save report or explicitly pass `id` to get a url')
        base = urlparse(_get_api().client.app_url)
        title = self.title.replace(' ', '-')
        scheme = base.scheme
        netloc = base.netloc
        path = os.path.join(self.entity, self.project, 'reports', f'{title}--{self.id}')
        params = ''
        query = ''
        fragment = ''
        return urlunparse((scheme, netloc, path, params, query, fragment))

    def save(self, draft: bool=False, clone: bool=False):
        model = self.to_model()
        projects = _get_api().projects(self.entity)
        is_new_project = True
        for p in projects:
            if p.name == self.project:
                is_new_project = False
                break
        if is_new_project:
            _get_api().create_project(self.project, self.entity)
        r = _get_api().client.execute(gql.upsert_view, variable_values={'id': None if clone or not model.id else model.id, 'name': internal._generate_name() if clone or not model.name else model.name, 'entityName': model.project.entity_name, 'projectName': model.project.name, 'description': model.description, 'displayName': model.display_name, 'type': 'runs/draft' if draft else 'runs', 'spec': model.spec.model_dump_json(by_alias=True, exclude_none=True)})
        viewspec = r['upsertView']['view']
        new_model = internal.ReportViewspec.model_validate(viewspec)
        self.id = new_model.id
        wandb.termlog(f'Saved report to: {self.url}')
        return self

    @classmethod
    def from_url(cls, url, *, as_model: bool=False):
        vs = _url_to_viewspec(url)
        model = internal.ReportViewspec.model_validate(vs)
        if as_model:
            return model
        return cls.from_model(model)

    def to_html(self, height: int=1024, hidden: bool=False) -> str:
        """Generate HTML containing an iframe displaying this report."""
        try:
            url = self.url + '?jupyter=true'
            style = f'border:none;width:100%;height:{height}px;'
            prefix = ''
            if hidden:
                style += 'display:none;'
                prefix = wandb.sdk.lib.ipython.toggle_button('report')
            return prefix + f'<iframe src={url!r} style={style!r}></iframe>'
        except AttributeError:
            wandb.termlog('HTML repr will be available after you save the report!')

    def _repr_html_(self) -> str:
        return self.to_html()