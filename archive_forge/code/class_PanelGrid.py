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
class PanelGrid(Block):
    runsets: LList['Runset'] = Field(default_factory=lambda: [Runset()])
    panels: LList['PanelTypes'] = Field(default_factory=list)
    active_runset: int = 0
    custom_run_colors: Dict[Union[RunId, RunsetGroup], str] = Field(default_factory=dict)
    _ref: Optional[internal.Ref] = Field(default_factory=lambda: None, init=False, repr=False)
    _open_viz: bool = Field(default_factory=lambda: True, init=False, repr=False)
    _panel_bank_sections: LList[dict] = Field(default_factory=list, init=False, repr=False)
    _panel_grid_metadata_ref: Optional[internal.Ref] = Field(default_factory=lambda: None, init=False, repr=False)

    def to_model(self):
        return internal.PanelGrid(metadata=internal.PanelGridMetadata(run_sets=[rs.to_model() for rs in self.runsets], panel_bank_section_config=internal.PanelBankSectionConfig(panels=[p.to_model() for p in self.panels]), panels=internal.PanelGridMetadataPanels(ref=self._panel_grid_metadata_ref, panel_bank_config=internal.PanelBankConfig(), open_viz=self._open_viz), custom_run_colors=_to_color_dict(self.custom_run_colors, self.runsets)))

    @classmethod
    def from_model(cls, model: internal.PanelGrid):
        runsets = [Runset.from_model(rs) for rs in model.metadata.run_sets]
        obj = cls(runsets=runsets, panels=[_lookup_panel(p) for p in model.metadata.panel_bank_section_config.panels], active_runset=model.metadata.open_run_set, custom_run_colors=_from_color_dict(model.metadata.custom_run_colors, runsets))
        obj._open_viz = model.metadata.open_viz
        obj._ref = model.metadata.panels.ref
        return obj

    @validator('panels')
    def _resolve_collisions(cls, v):
        v2 = _resolve_collisions(v)
        return v2

    @validator('runsets')
    def _validate_list_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError('must have at least one runset')
        return v