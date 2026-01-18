from __future__ import annotations
import sys
from . import config
from . import exclusions
from .. import event
from .. import schema
from .. import types as sqltypes
from ..orm import mapped_column as _orm_mapped_column
from ..util import OrderedDict
def _schema_column(factory, args, kw):
    test_opts = {k: kw.pop(k) for k in list(kw) if k.startswith('test_')}
    if not config.requirements.foreign_key_ddl.enabled_for_config(config):
        args = [arg for arg in args if not isinstance(arg, schema.ForeignKey)]
    construct = factory(*args, **kw)
    if factory is schema.Column:
        col = construct
    else:
        col = construct.column
    if test_opts.get('test_needs_autoincrement', False) and kw.get('primary_key', False):
        if col.default is None and col.server_default is None:
            col.autoincrement = True
        col.info['test_needs_autoincrement'] = True
        if exclusions.against(config._current, 'oracle'):

            def add_seq(c, tbl):
                c._init_items(schema.Sequence(_truncate_name(config.db.dialect, tbl.name + '_' + c.name + '_seq'), optional=True))
            event.listen(col, 'after_parent_attach', add_seq, propagate=True)
    return construct