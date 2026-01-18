import html
import json
from typing import Any, Dict, List, Optional
from IPython import get_ipython
from IPython.core.magic import Magics, cell_magic, magics_class, needs_local_scope
from IPython.display import HTML, display
from triad import ParamDict
from triad.utils.convert import to_instance
from triad.utils.pyarrow import _field_to_expression
from fugue import DataFrame, DataFrameDisplay, ExecutionEngine
from fugue import fsql as fugue_sql
from fugue import get_dataset_display, make_execution_engine
from fugue.dataframe import YieldedDataFrame
from fugue.exceptions import FugueSQLSyntaxError
class JupyterDataFrameDisplay(DataFrameDisplay):

    def show(self, n: int=10, with_count: bool=False, title: Optional[str]=None) -> None:
        components: List[Any] = []
        if title is not None:
            components.append(HTML(f'<h3>{html.escape(title)}</h3>'))
        if with_count:
            count = self.df.count()
        else:
            count = -1
        components.append(HTML(self._generate_df_html(n)))
        if count >= 0:
            components.append(HTML(f'<strong>total count: {count}</strong>'))
        display(*components)

    def repr_html(self) -> str:
        return self._generate_df_html(10)

    def _generate_df_html(self, n: int) -> str:
        res: List[str] = []
        pdf = self.df.head(n).as_pandas()
        cols = [_field_to_expression(f) for f in self.df.schema.fields]
        pdf.columns = cols
        res.append(pdf._repr_html_())
        schema = type(self.df).__name__ + ': ' + str(self.df.schema)
        res.append('<font size="-1">' + html.escape(schema) + '</font>')
        return '\n'.join(res)