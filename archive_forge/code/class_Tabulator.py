from __future__ import annotations
import datetime as dt
import uuid
from functools import partial
from types import FunctionType, MethodType
from typing import (
import numpy as np
import param
from bokeh.model import Model
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from bokeh.models.widgets.tables import (
from bokeh.util.serialization import convert_datetime_array
from pyviz_comms import JupyterComm
from ..depends import transform_reference
from ..io.resources import CDN_DIST, CSS_URLS
from ..io.state import state
from ..reactive import Reactive, ReactiveData
from ..util import (
from ..util.warnings import warn
from .base import Widget
from .button import Button
from .input import TextInput
class Tabulator(BaseTable):
    """
    The `Tabulator` widget wraps the [Tabulator js](http://tabulator.info/)
    table to provide a full-featured, very powerful interactive table.

    Reference: https://panel.holoviz.org/reference/widgets/Tabulator.html

    :Example:

    >>> Tabulator(df, theme='site', pagination='remote', page_size=25)
    """
    buttons = param.Dict(default={}, nested_refs=True, doc='\n        Dictionary mapping from column name to a HTML element\n        to use as the button icon.')
    expanded = param.List(default=[], nested_refs=True, doc='\n        List of expanded rows, only applicable if a row_content function\n        has been defined.')
    embed_content = param.Boolean(default=False, doc='\n        Whether to embed the row_content or render it dynamically\n        when a row is expanded.')
    filters = param.List(default=[], doc="\n        List of client-side filters declared as dictionaries containing\n        'field', 'type' and 'value' keys.")
    frozen_columns = param.ClassSelector(class_=(list, dict), default=[], nested_refs=True, doc="\n        One of:\n        - List indicating the columns to freeze. The column(s) may be\n        selected by name or index.\n        - Dict indicating columns to freeze as keys and their freeze location\n        as values, freeze location is either 'right' or 'left'.")
    frozen_rows = param.List(default=[], nested_refs=True, doc='\n        List indicating the rows to freeze. If set, the\n        first N rows will be frozen, which prevents them from scrolling\n        out of frame; if set to a negative value the last N rows will be\n        frozen.')
    groups = param.Dict(default={}, nested_refs=True, doc='\n        Dictionary mapping defining the groups.')
    groupby = param.List(default=[], nested_refs=True, doc='\n        Groups rows in the table by one or more columns.')
    header_align = param.ClassSelector(default={}, nested_refs=True, class_=(dict, str), doc="\n        A mapping from column name to alignment or a fixed column\n        alignment, which should be one of 'left', 'center', 'right'.")
    header_filters = param.ClassSelector(class_=(bool, dict), nested_refs=True, doc='\n        Whether to enable filters in the header or dictionary\n        configuring filters for each column.')
    hidden_columns = param.List(default=[], nested_refs=True, doc='\n        List of columns to hide.')
    layout = param.ObjectSelector(default='fit_data_table', objects=['fit_data', 'fit_data_fill', 'fit_data_stretch', 'fit_data_table', 'fit_columns'])
    pagination = param.ObjectSelector(default=None, allow_None=True, objects=['local', 'remote'])
    page = param.Integer(default=1, doc='\n        Currently selected page (indexed starting at 1), if pagination is enabled.')
    page_size = param.Integer(default=20, bounds=(1, None), doc='\n        Number of rows to render per page, if pagination is enabled.')
    row_content = param.Callable(doc='\n        A function which is given the DataFrame row and should return\n        a Panel object to render as additional detail below the row.')
    row_height = param.Integer(default=30, doc='\n        The height of each table row.')
    selectable = param.ClassSelector(default=True, class_=(bool, str, int), doc="\n        Defines the selection mode of the Tabulator.\n\n          - True\n              Selects rows on click. To select multiple use Ctrl-select,\n              to select a range use Shift-select\n          - False\n              Disables selection\n          - 'checkbox'\n              Adds a column of checkboxes to toggle selections\n          - 'checkbox-single'\n              Same as 'checkbox' but header does not allow select/deselect all\n          - 'toggle'\n              Selection toggles when clicked\n          - int\n              The maximum number of selectable rows.\n        ")
    selectable_rows = param.Callable(default=None, doc='\n        A function which given a DataFrame should return a list of\n        rows by integer index, which are selectable.')
    sortable = param.ClassSelector(default=True, class_=(bool, dict), doc='\n        Whether the columns in the table should be sortable.\n        Can either be specified as a simple boolean toggling the behavior\n        on and off or as a dictionary specifying the option per column.')
    theme = param.ObjectSelector(default='simple', objects=['default', 'site', 'simple', 'midnight', 'modern', 'bootstrap', 'bootstrap4', 'materialize', 'bulma', 'semantic-ui', 'fast', 'bootstrap5'], doc='\n        Tabulator CSS theme to apply to table.')
    theme_classes = param.List(default=[], nested_refs=True, item_type=str, doc='\n       List of extra CSS classes to apply to the Tabulator element\n       to customize the theme.')
    title_formatters = param.Dict(default={}, nested_refs=True, doc='\n       Tabulator formatter specification to use for a particular column\n       header title.')
    _data_params: ClassVar[List[str]] = ['value', 'page', 'page_size', 'pagination', 'sorters', 'filters']
    _config_params: ClassVar[List[str]] = ['frozen_columns', 'groups', 'selectable', 'hierarchical', 'sortable']
    _content_params: ClassVar[List[str]] = _data_params + ['expanded', 'row_content', 'embed_content']
    _manual_params: ClassVar[List[str]] = BaseTable._manual_params + _config_params
    _priority_changes: ClassVar[List[str]] = ['data']
    _rename: ClassVar[Mapping[str, str | None]] = {'selection': None, 'row_content': None, 'row_height': None, 'text_align': None, 'embed_content': None, 'header_align': None, 'header_filters': None, 'styles': 'cell_styles', 'title_formatters': None, 'sortable': None}
    _MAX_ROW_LIMITS: ClassVar[Tuple[int, int]] = (200, 10000)
    _stylesheets = [CSS_URLS['font-awesome']]

    def __init__(self, value=None, **params):
        import pandas.io.formats.style
        click_handler = params.pop('on_click', None)
        edit_handler = params.pop('on_edit', None)
        if isinstance(value, pandas.io.formats.style.Styler):
            style = value
            value = value.data
        else:
            style = None
        configuration = params.pop('configuration', {})
        self.style = None
        self._computed_styler = None
        self._child_panels = {}
        self._explicit_pagination = 'pagination' in params
        self._on_edit_callbacks = []
        self._on_click_callbacks = {}
        self._old_value = None
        super().__init__(value=value, **params)
        self._configuration = configuration
        self.param.watch(self._update_children, self._content_params)
        if click_handler:
            self.on_click(click_handler)
        if edit_handler:
            self.on_edit(edit_handler)
        if style is not None:
            self.style._todo = style._todo

    @param.depends('value', watch=True, on_init=True)
    def _apply_max_size(self):
        """
        Ensure large tables automatically enable remote pagination.
        """
        if self.value is None or self._explicit_pagination:
            return
        with param.parameterized.discard_events(self):
            if self.hierarchical:
                pass
            elif self._MAX_ROW_LIMITS[0] < len(self.value) <= self._MAX_ROW_LIMITS[1]:
                self.pagination = 'local'
            elif len(self.value) > self._MAX_ROW_LIMITS[1]:
                self.pagination = 'remote'
        self._explicit_pagination = False

    @param.depends('pagination', watch=True)
    def _set_explicict_pagination(self):
        self._explicit_pagination = True

    @staticmethod
    def _validate_iloc(idx, iloc):
        if not isinstance(iloc, int):
            raise ValueError(f'The Tabulator widget expects the provided `value` Pandas DataFrame to have unique indexes, in particular when it has to deal with click or edit events. Found this duplicate index: {idx!r}')

    def _validate(self, *events):
        super()._validate(*events)
        if self.value is not None:
            todo = []
            if self.style is not None:
                todo = self.style._todo
            try:
                self.style = self.value.style
                self.style._todo = todo
            except Exception:
                pass

    def _cleanup(self, root: Model | None=None) -> None:
        for p in self._child_panels.values():
            p._cleanup(root)
        super()._cleanup(root)

    def _process_event(self, event) -> None:
        if event.event_name == 'selection-change':
            self._update_selection(event)
            return
        event_col = self._renamed_cols.get(event.column, event.column)
        if self.pagination == 'remote':
            nrows = self.page_size
            event.row = event.row + (self.page - 1) * nrows
        idx = self._index_mapping.get(event.row, event.row)
        iloc = self.value.index.get_loc(idx)
        self._validate_iloc(idx, iloc)
        event.row = iloc
        if event_col not in self.buttons:
            if event_col not in self.value.columns:
                event.value = self.value.index[event.row]
            else:
                event.value = self.value[event_col].iloc[event.row]
        if event.event_name == 'table-edit':
            if event.pre:
                import pandas as pd
                filter_df = pd.DataFrame({event.column: [event.value]})
                filters = self._get_header_filters(filter_df)
                if filters and filters[0].any():
                    self._edited_indexes.append(idx)
            else:
                if self._old_value is not None:
                    event.old = self._old_value[event_col].iloc[event.row]
                for cb in self._on_edit_callbacks:
                    state.execute(partial(cb, event), schedule=False)
                self._update_style()
        else:
            for cb in self._on_click_callbacks.get(None, []):
                state.execute(partial(cb, event), schedule=False)
            for cb in self._on_click_callbacks.get(event_col, []):
                state.execute(partial(cb, event), schedule=False)

    def _get_theme(self, theme, resources=None):
        from ..models.tabulator import _TABULATOR_THEMES_MAPPING, THEME_PATH
        theme_ = _TABULATOR_THEMES_MAPPING.get(theme, theme)
        fname = 'tabulator' if theme_ == 'default' else f'tabulator_{theme_}'
        theme_url = f'{CDN_DIST}bundled/datatabulator/{THEME_PATH}{fname}.min.css'
        if self._widget_type is not None:
            self._widget_type.__css_raw__ = [theme_url]
        return theme_url

    def _update_columns(self, event, model):
        if event.name not in self._config_params:
            super()._update_columns(event, model)
            if event.name in ('editors', 'formatters', 'sortable') and (not any((isinstance(v, (str, dict)) for v in event.new.values()))):
                return
        model.configuration = self._get_configuration(model.columns)

    def _process_data(self, data):
        self._old_value = self.value.copy()
        import pandas as pd
        df = pd.DataFrame(data)
        filters = self._get_header_filters(df)
        if filters:
            mask = filters[0]
            for f in filters:
                mask &= f
            if self._edited_indexes:
                edited_mask = df[self.value.index.name or 'index'].isin(self._edited_indexes)
                mask = mask | edited_mask
            df = df[mask]
        data = {col: df[col].values for col in df.columns}
        return super()._process_data(data)

    def _get_data(self):
        if self.pagination != 'remote' or self.value is None:
            return super()._get_data()
        import pandas as pd
        df = self._filter_dataframe(self.value)
        df = self._sort_df(df)
        nrows = self.page_size
        start = (self.page - 1) * nrows
        page_df = df.iloc[start:start + nrows]
        if isinstance(self.value.index, pd.MultiIndex):
            indexes = [f'level_{i}' if n is None else n for i, n in enumerate(df.index.names)]
        else:
            default_index = 'level_0' if 'index' in df.columns else 'index'
            indexes = [df.index.name or default_index]
        if len(indexes) > 1:
            page_df = page_df.reset_index()
        data = ColumnDataSource.from_df(page_df).items()
        return (df, {k if isinstance(k, str) else str(k): v for k, v in data})

    def _get_style_data(self, recompute=True):
        if self.value is None or self.style is None or self.value.empty:
            return {}
        df = self._processed
        if recompute:
            try:
                self._computed_styler = styler = df.style
            except Exception:
                self._computed_styler = None
                return {}
            if styler is None:
                return {}
            styler._todo = styler_update(self.style, df)
            try:
                styler._compute()
            except Exception:
                styler._todo = []
        else:
            styler = self._computed_styler
        if styler is None:
            return {}
        offset = 1 + len(self.indexes) + int(self.selectable in ('checkbox', 'checkbox-single')) + int(bool(self.row_content))
        if self.pagination == 'remote':
            start = (self.page - 1) * self.page_size
            end = start + self.page_size
        column_mapper = {}
        frozen_cols = self.frozen_columns
        column_mapper = {}
        if isinstance(frozen_cols, list):
            nfrozen = len(frozen_cols)
            non_frozen = [col for col in df.columns if col not in frozen_cols]
            for i, col in enumerate(df.columns):
                if col in frozen_cols:
                    column_mapper[i] = frozen_cols.index(col) - len(self.indexes)
                else:
                    column_mapper[i] = nfrozen + non_frozen.index(col)
        elif isinstance(frozen_cols, dict):
            left_cols = [col for col, p in frozen_cols.items() if p in 'left']
            right_cols = [col for col, p in frozen_cols.items() if p in 'right']
            non_frozen = [col for col in df.columns if col not in frozen_cols]
            for i, col in enumerate(df.columns):
                if col in left_cols:
                    column_mapper[i] = left_cols.index(col) - len(self.indexes)
                elif col in right_cols:
                    column_mapper[i] = len(left_cols) + len(non_frozen) + right_cols.index(col)
                else:
                    column_mapper[i] = len(left_cols) + non_frozen.index(col)
        styles = {}
        for (r, c), s in styler.ctx.items():
            if self.pagination == 'remote':
                if r < start or r >= end:
                    continue
                else:
                    r -= start
            if r not in styles:
                styles[int(r)] = {}
            c = column_mapper.get(int(c), int(c))
            styles[int(r)][offset + c] = s
        return {'id': uuid.uuid4().hex, 'data': styles}

    def _get_selectable(self):
        if self.value is None or self.selectable_rows is None:
            return None
        df = self._processed
        if self.pagination == 'remote':
            nrows = self.page_size
            start = (self.page - 1) * nrows
            df = df.iloc[start:start + nrows]
        return self.selectable_rows(df)

    def _update_style(self, recompute=True):
        styles = self._get_style_data(recompute)
        msg = {'cell_styles': styles}
        for ref, (m, _) in self._models.items():
            self._apply_update([], msg, m, ref)

    def _get_children(self, old={}):
        if self.row_content is None or self.value is None:
            return {}
        from ..pane import panel
        df = self._processed
        if self.pagination == 'remote':
            nrows = self.page_size
            start = (self.page - 1) * nrows
            df = df.iloc[start:start + nrows]
        children = {}
        for i in range(len(df)) if self.embed_content else self.expanded:
            if i in old:
                children[i] = old[i]
            else:
                children[i] = panel(self.row_content(df.iloc[i]))
        return children

    def _get_model_children(self, panels, doc, root, parent, comm=None):
        ref = root.ref['id']
        models = {}
        for i, p in panels.items():
            if ref in p._models:
                model = p._models[ref][0]
            else:
                model = p._get_model(doc, root, parent, comm)
            model.margin = (0, 0, 0, 0)
            models[i] = model
        return models

    def _indexes_changed(self, old, new):
        """
        Comparator that checks whether DataFrame indexes have changed.

        If indexes and length are unchanged we assume we do not
        have to reset various settings including expanded rows,
        scroll position, pagination etc.
        """
        if type(old) != type(new) or isinstance(new, dict):
            return True
        elif len(old) != len(new):
            return False
        return (old.index != new.index).any()

    def _update_children(self, *events):
        cleanup, reuse = (set(), set())
        page_events = ('page', 'page_size', 'pagination')
        for event in events:
            if event.name == 'expanded' and len(events) == 1:
                cleanup = set(event.old) - set(event.new)
                reuse = set(event.old) & set(event.new)
            elif event.name == 'value' and self._indexes_changed(event.old, event.new) or (event.name in page_events and (not self._updating)) or (self.pagination == 'remote' and event.name == 'sorters'):
                self.expanded = []
                return
        old_panels = self._child_panels
        self._child_panels = child_panels = self._get_children({i: old_panels[i] for i in reuse})
        for ref, (m, _) in self._models.items():
            root, doc, comm = state._views[ref][1:]
            for idx in cleanup:
                old_panels[idx]._cleanup(root)
            children = self._get_model_children(child_panels, doc, root, m, comm)
            msg = {'children': children}
            self._apply_update([], msg, m, ref)

    @updating
    def _stream(self, stream, rollover=None, follow=True):
        if self.pagination == 'remote':
            length = self._length
            nrows = self.page_size
            max_page = max(length // nrows + bool(length % nrows), 1)
            if self.page != max_page:
                return
        super()._stream(stream, rollover)
        self._update_style()
        self._update_selectable()
        self._update_index_mapping()

    def stream(self, stream_value, rollover=None, reset_index=True, follow=True):
        for ref, (model, _) in self._models.items():
            self._apply_update([], {'follow': follow}, model, ref)
        if follow and self.pagination:
            length = self._length
            nrows = self.page_size
            self.page = max(length // nrows + bool(length % nrows), 1)
        super().stream(stream_value, rollover, reset_index)
        if follow and self.pagination:
            self._update_max_page()

    @updating
    def _patch(self, patch):
        if self.filters or self.sorters:
            self._updating = False
            self._update_cds()
            return
        if self.pagination == 'remote':
            nrows = self.page_size
            start = (self.page - 1) * nrows
            end = start + nrows
            filtered = {}
            for c, values in patch.items():
                values = [(ind, val) for ind, val in values if ind >= start and ind < end]
                if values:
                    filtered[c] = values
            patch = filtered
        if not patch:
            return
        super()._patch(patch)
        self._update_style()
        self._update_selectable()

    def _update_cds(self, *events):
        if any((event.name == 'filters' for event in events)):
            self._edited_indexes = []
        page_events = ('page', 'page_size', 'sorters', 'filters')
        if self._updating:
            return
        elif events and all((e.name in page_events[:-1] for e in events)) and (self.pagination == 'local'):
            return
        elif events and all((e.name in page_events for e in events)) and (not self.pagination):
            self._processed, _ = self._get_data()
            return
        elif self.pagination == 'remote':
            self._processed = None
        recompute = not all((e.name in ('page', 'page_size', 'pagination') for e in events))
        super()._update_cds(*events)
        if self.pagination:
            self._update_max_page()
            self._update_selected()
        self._update_style(recompute)
        self._update_selectable()

    def _update_selectable(self):
        selectable = self._get_selectable()
        for ref, (model, _) in self._models.items():
            self._apply_update([], {'selectable_rows': selectable}, model, ref)

    def _update_max_page(self):
        length = self._length
        nrows = self.page_size
        max_page = max(length // nrows + bool(length % nrows), 1)
        self.param.page.bounds = (1, max_page)
        for ref, (model, _) in self._models.items():
            self._apply_update([], {'max_page': max_page}, model, ref)

    def _update_selected(self, *events: param.parameterized.Event, indices=None):
        kwargs = {}
        if self.pagination == 'remote' and self.value is not None:
            index = self.value.iloc[self.selection].index
            indices = []
            for ind in index.values:
                try:
                    iloc = self._processed.index.get_loc(ind)
                    self._validate_iloc(ind, iloc)
                    indices.append((ind, iloc))
                except KeyError:
                    continue
            nrows = self.page_size
            start = (self.page - 1) * nrows
            end = start + nrows
            p_range = self._processed.index[start:end]
            kwargs['indices'] = [iloc - start for ind, iloc in indices if ind in p_range]
        super()._update_selected(*events, **kwargs)

    def _update_column(self, column: str, array: np.ndarray):
        import pandas as pd
        if self.pagination != 'remote':
            index = self._processed.index.values
            self.value.loc[index, column] = array
            with pd.option_context('mode.chained_assignment', None):
                self._processed[column] = array
            return
        nrows = self.page_size
        start = (self.page - 1) * nrows
        end = start + nrows
        index = self._processed.iloc[start:end].index.values
        self.value.loc[index, column] = array
        with pd.option_context('mode.chained_assignment', None):
            self._processed.loc[index, column] = array

    def _update_selection(self, indices: List[int] | SelectionEvent):
        if self.pagination != 'remote':
            self.selection = indices
            return
        if isinstance(indices, list):
            selected = True
            ilocs = []
        else:
            selected = indices.selected
            ilocs = [] if indices.flush else self.selection.copy()
            indices = indices.indices
        nrows = self.page_size
        start = (self.page - 1) * nrows
        index = self._processed.iloc[[start + ind for ind in indices]].index
        for v in index.values:
            try:
                iloc = self.value.index.get_loc(v)
                self._validate_iloc(v, iloc)
            except KeyError:
                continue
            if selected:
                ilocs.append(iloc)
            elif iloc in ilocs:
                ilocs.remove(iloc)
        ilocs = list(dict.fromkeys(ilocs))
        if isinstance(self.selectable, int) and (not isinstance(self.selectable, bool)):
            ilocs = ilocs[len(ilocs) - self.selectable:]
        self.selection = ilocs

    def _get_properties(self, doc: Document) -> Dict[str, Any]:
        properties = super()._get_properties(doc)
        properties['configuration'] = self._get_configuration(properties['columns'])
        properties['cell_styles'] = self._get_style_data()
        properties['indexes'] = self.indexes
        if self.pagination:
            length = self._length
            properties['max_page'] = max(length // self.page_size + bool(length % self.page_size), 1)
        if isinstance(self.selectable, str) and self.selectable.startswith('checkbox'):
            properties['select_mode'] = 'checkbox'
        else:
            properties['select_mode'] = self.selectable
        return properties

    def _process_param_change(self, params):
        if 'theme' in params or 'stylesheets' in params:
            theme_url = self._get_theme(params.pop('theme', self.theme))
            params['stylesheets'] = params.get('stylesheets', self.stylesheets) + [ImportedStyleSheet(url=theme_url)]
        params = Reactive._process_param_change(self, params)
        if 'disabled' in params:
            params['editable'] = not params.pop('disabled') and len(self.indexes) <= 1
        if 'frozen_rows' in params:
            length = self._length
            params['frozen_rows'] = [length + r if r < 0 else r for r in params['frozen_rows']]
        if 'hidden_columns' in params:
            import pandas as pd
            if not self.show_index and self.value is not None and (not isinstance(self.value.index, pd.MultiIndex)):
                params['hidden_columns'] = params['hidden_columns'] + [self.value.index.name or 'index']
        if 'selectable_rows' in params:
            params['selectable_rows'] = self._get_selectable()
        return params

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        Tabulator._widget_type = lazy_load('panel.models.tabulator', 'DataTabulator', isinstance(comm, JupyterComm), root)
        model = super()._get_model(doc, root, parent, comm)
        root = root or model
        self._child_panels = child_panels = self._get_children()
        model.children = self._get_model_children(child_panels, doc, root, parent, comm)
        self._link_props(model, ['page', 'sorters', 'expanded', 'filters'], doc, root, comm)
        self._register_events('cell-click', 'table-edit', 'selection-change', model=model, doc=doc, comm=comm)
        return model

    def _get_filter_spec(self, column: TableColumn) -> Dict[str, Any]:
        fspec = {}
        if not self.header_filters or (isinstance(self.header_filters, dict) and column.field not in self.header_filters):
            return fspec
        elif self.header_filters == True:
            if column.field in self.indexes:
                if len(self.indexes) == 1:
                    col = self.value.index
                else:
                    col = self.value.index.get_level_values(self.indexes.index(column.field))
                if col.dtype.kind in 'uif':
                    fspec['headerFilter'] = 'number'
                elif col.dtype.kind == 'b':
                    fspec['headerFilter'] = 'tickCross'
                    fspec['headerFilterParams'] = {'tristate': True, 'indeterminateValue': None}
                elif isdatetime(col) or col.dtype.kind == 'M':
                    fspec['headerFilter'] = False
                else:
                    fspec['headerFilter'] = True
            elif isinstance(column.editor, DateEditor):
                fspec['headerFilter'] = False
            else:
                fspec['headerFilter'] = True
            return fspec
        filter_type = self.header_filters[column.field]
        if isinstance(filter_type, dict):
            filter_params = dict(filter_type)
            filter_type = filter_params.pop('type', True)
            filter_func = filter_params.pop('func', None)
            filter_placeholder = filter_params.pop('placeholder', None)
        else:
            filter_params = {}
            filter_func = None
            filter_placeholder = None
        if filter_type in ['select', 'autocomplete']:
            self.param.warning(f'The {filter_type!r} filter has been deprecated, use instead the "list" filter type to configure column {column.field!r}')
            filter_type = 'list'
            if filter_params.get('values', False) is True:
                self.param.warning(f'Setting "values" to True has been deprecated, instead set "valuesLookup" to True to configure column {column.field!r}')
                del filter_params['values']
                filter_params['valuesLookup'] = True
        if filter_type == 'list' and (not filter_params):
            filter_params = {'valuesLookup': True}
        fspec['headerFilter'] = filter_type
        if filter_params:
            fspec['headerFilterParams'] = filter_params
        if filter_func:
            fspec['headerFilterFunc'] = filter_func
        if filter_placeholder:
            fspec['headerFilterPlaceholder'] = filter_placeholder
        return fspec

    def _config_columns(self, column_objs: List[TableColumn]) -> List[Dict[str, Any]]:
        column_objs = list(column_objs)
        groups = {}
        columns = []
        selectable = self.selectable
        if self.row_content:
            columns.append({'formatter': 'expand'})
        if isinstance(selectable, str) and selectable.startswith('checkbox'):
            title = '' if selectable.endswith('-single') else 'rowSelection'
            columns.append({'formatter': 'rowSelection', 'titleFormatter': title, 'hozAlign': 'center', 'headerSort': False, 'frozen': True, 'width': 40})
        if isinstance(self.frozen_columns, dict):
            left_frozen_columns = [col for col in column_objs if self.frozen_columns.get(col.field, self.frozen_columns.get(column_objs.index(col))) == 'left']
            right_frozen_columns = [col for col in column_objs if self.frozen_columns.get(col.field, self.frozen_columns.get(column_objs.index(col))) == 'right']
            non_frozen_columns = [col for col in column_objs if col.field not in self.frozen_columns and column_objs.index(col) not in self.frozen_columns]
            ordered_columns = left_frozen_columns + non_frozen_columns + right_frozen_columns
        else:
            ordered_columns = []
            for col in self.frozen_columns:
                if isinstance(col, int):
                    ordered_columns.append(column_objs.pop(col))
                else:
                    cols = [c for c in column_objs if c.field == col]
                    if cols:
                        ordered_columns.append(cols[0])
                        column_objs.remove(cols[0])
            ordered_columns += column_objs
        grouping = {group: [str(gc) for gc in group_cols] for group, group_cols in self.groups.items()}
        for i, column in enumerate(ordered_columns):
            field = column.field
            matching_groups = [group for group, group_cols in grouping.items() if field in group_cols]
            col_dict = dict(field=field)
            if isinstance(self.sortable, dict):
                col_dict['headerSort'] = self.sortable.get(field, True)
            elif not self.sortable:
                col_dict['headerSort'] = self.sortable
            if isinstance(self.text_align, str):
                col_dict['hozAlign'] = self.text_align
            elif field in self.text_align:
                col_dict['hozAlign'] = self.text_align[field]
            if isinstance(self.header_align, str):
                col_dict['headerHozAlign'] = self.header_align
            elif field in self.header_align:
                col_dict['headerHozAlign'] = self.header_align[field]
            formatter = self.formatters.get(field)
            if isinstance(formatter, str):
                col_dict['formatter'] = formatter
            elif isinstance(formatter, dict):
                formatter = dict(formatter)
                col_dict['formatter'] = formatter.pop('type')
                col_dict['formatterParams'] = formatter
            title_formatter = self.title_formatters.get(field)
            if isinstance(title_formatter, str):
                col_dict['titleFormatter'] = title_formatter
            elif isinstance(title_formatter, dict):
                title_formatter = dict(title_formatter)
                col_dict['titleFormatter'] = title_formatter.pop('type')
                col_dict['titleFormatterParams'] = title_formatter
            col_name = self._renamed_cols[field]
            if field in self.indexes:
                if len(self.indexes) == 1:
                    dtype = self.value.index.dtype
                else:
                    dtype = self.value.index.get_level_values(self.indexes.index(field)).dtype
            else:
                dtype = self.value.dtypes[col_name]
            if dtype.kind == 'M':
                col_dict['sorter'] = 'timestamp'
            elif dtype.kind in 'iuf':
                col_dict['sorter'] = 'number'
            elif dtype.kind == 'b':
                col_dict['sorter'] = 'boolean'
            editor = self.editors.get(field)
            if field in self.editors and editor is None:
                col_dict['editable'] = False
            if isinstance(editor, str):
                col_dict['editor'] = editor
            elif isinstance(editor, dict):
                editor = dict(editor)
                col_dict['editor'] = editor.pop('type')
                col_dict['editorParams'] = editor
            if col_dict.get('editor') in ['select', 'autocomplete']:
                self.param.warning(f'The {col_dict['editor']!r} editor has been deprecated, use instead the "list" editor type to configure column {field!r}')
                col_dict['editor'] = 'list'
                if col_dict.get('editorParams', {}).get('values', False) is True:
                    del col_dict['editorParams']['values']
                    col_dict['editorParams']['valuesLookup'] = True
            if field in self.frozen_columns or i in self.frozen_columns:
                col_dict['frozen'] = True
            if isinstance(self.widths, dict) and isinstance(self.widths.get(field), str):
                col_dict['width'] = self.widths[field]
            col_dict.update(self._get_filter_spec(column))
            if matching_groups:
                group = matching_groups[0]
                if group in groups:
                    groups[group]['columns'].append(col_dict)
                    continue
                group_dict = {'title': group, 'columns': [col_dict]}
                groups[group] = group_dict
                columns.append(group_dict)
            else:
                columns.append(col_dict)
        return columns

    def _get_configuration(self, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns the Tabulator configuration.
        """
        configuration = dict(self._configuration)
        if 'selectable' not in configuration:
            configuration['selectable'] = self.selectable
        if self.groups and 'columns' in configuration:
            raise ValueError('Groups must be defined either explicitly or via the configuration, not both.')
        configuration['columns'] = self._config_columns(columns)
        configuration['dataTree'] = self.hierarchical
        if self.sizing_mode in ('stretch_height', 'stretch_both'):
            configuration['maxHeight'] = '100%'
        elif self.height:
            configuration['height'] = self.height
        return configuration

    def download(self, filename: str='table.csv'):
        """
        Triggers downloading of the table as a CSV or JSON.

        Arguments
        ---------
        filename: str
            The filename to save the table as.
        """
        for ref, (model, _) in self._models.items():
            self._apply_update({}, {'filename': filename}, model, ref)
            self._apply_update({}, {'download': not model.download}, model, ref)

    def download_menu(self, text_kwargs={}, button_kwargs={}):
        """
        Returns a menu containing a TextInput and Button widget to set
        the filename and trigger a client-side download of the data.

        Arguments
        ---------
        text_kwargs: dict
            Keyword arguments passed to the TextInput constructor
        button_kwargs: dict
            Keyword arguments passed to the Button constructor

        Returns
        -------
        filename: TextInput
            The TextInput widget setting a filename.
        button: Button
            The Button that triggers a download.
        """
        text_kwargs = dict(text_kwargs)
        if 'name' not in text_kwargs:
            text_kwargs['name'] = 'Filename'
        if 'value' not in text_kwargs:
            text_kwargs['value'] = 'table.csv'
        filename = TextInput(**text_kwargs)
        button_kwargs = dict(button_kwargs)
        if 'name' not in button_kwargs:
            button_kwargs['name'] = 'Download'
        button = Button(**button_kwargs)
        button.js_on_click({'table': self, 'filename': filename}, code='\n        table.filename = filename.value\n        table.download = !table.download\n        ')
        return (filename, button)

    def on_edit(self, callback: Callable[[TableEditEvent], None]):
        """
        Register a callback to be executed when a cell is edited.
        Whenever a cell is edited on_edit callbacks are called with
        a TableEditEvent as the first argument containing the column,
        row and value of the edited cell.

        Arguments
        ---------
        callback: (callable)
            The callback to run on edit events.
        """
        self._on_edit_callbacks.append(callback)

    def on_click(self, callback: Callable[[CellClickEvent], None], column: Optional[str]=None):
        """
        Register a callback to be executed when any cell is clicked.
        The callback is given a CellClickEvent declaring the column
        and row of the cell that was clicked.

        Arguments
        ---------
        callback: (callable)
            The callback to run on edit events.
        column: (str)
            Optional argument restricting the callback to a specific
            column.
        """
        if column not in self._on_click_callbacks:
            self._on_click_callbacks[column] = []
        self._on_click_callbacks[column].append(callback)

    @property
    def current_view(self) -> pd.DataFrame:
        """
        Returns the current view of the table after filtering and
        sorting are applied.
        """
        df = self._processed
        if self.pagination == 'remote':
            return df
        return self._sort_df(df)