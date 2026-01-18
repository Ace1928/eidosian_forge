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
class BarPlot(Panel):
    title: Optional[str] = None
    metrics: LList[MetricType] = Field(default_factory=list)
    orientation: Literal['v', 'h'] = 'h'
    range_x: Range = Field(default_factory=lambda: (None, None))
    title_x: Optional[str] = None
    title_y: Optional[str] = None
    groupby: Optional[str] = None
    groupby_aggfunc: Optional[GroupAgg] = None
    groupby_rangefunc: Optional[GroupArea] = None
    max_runs_to_show: Optional[int] = None
    max_bars_to_show: Optional[int] = None
    custom_expressions: Optional[LList[str]] = None
    legend_template: Optional[str] = None
    font_size: Optional[FontSize] = None
    line_titles: Optional[dict] = None
    line_colors: Optional[dict] = None

    def to_model(self):
        obj = internal.BarPlot(config=internal.BarPlotConfig(chart_title=self.title, metrics=[_metric_to_backend(name) for name in _listify(self.metrics)], vertical=self.orientation == 'v', x_axis_min=self.range_x[0], x_axis_max=self.range_x[1], x_axis_title=self.title_x, y_axis_title=self.title_y, group_by=self.groupby, group_agg=self.groupby_aggfunc, group_area=self.groupby_rangefunc, limit=self.max_runs_to_show, bar_limit=self.max_bars_to_show, expressions=self.custom_expressions, legend_template=self.legend_template, font_size=self.font_size, override_series_titles=self.line_titles, override_colors=self.line_colors), layout=self.layout.to_model(), id=self.id)
        obj.ref = self._ref
        return obj

    @classmethod
    def from_model(cls, model: internal.ScatterPlot):
        obj = cls(title=model.config.chart_title, metrics=[_metric_to_frontend(name) for name in model.config.metrics], orientation='v' if model.config.vertical else 'h', range_x=(model.config.x_axis_min, model.config.x_axis_max), title_x=model.config.x_axis_title, title_y=model.config.y_axis_title, groupby=model.config.group_by, groupby_aggfunc=model.config.group_agg, groupby_rangefunc=model.config.group_area, max_runs_to_show=model.config.limit, max_bars_to_show=model.config.bar_limit, custom_expressions=model.config.expressions, legend_template=model.config.legend_template, font_size=model.config.font_size, line_titles=model.config.override_series_titles, line_colors=model.config.override_colors, layout=Layout.from_model(model.layout), id=model.id)
        obj._ref = model.ref
        return obj