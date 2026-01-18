from __future__ import annotations
from contextlib import contextmanager
from contextlib import nullcontext
import logging
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import ContextManager
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine
from sqlalchemy.engine import url as sqla_url
from sqlalchemy.engine.strategies import MockEngineStrategy
from .. import ddl
from .. import util
from ..util import sqla_compat
from ..util.compat import EncodedIO
def _delete_version(self, version: str) -> None:
    self.heads.remove(version)
    ret = self.context.impl._exec(self.context._version.delete().where(self.context._version.c.version_num == literal_column("'%s'" % version)))
    if not self.context.as_sql and self.context.dialect.supports_sane_rowcount and (ret is not None) and (ret.rowcount != 1):
        raise util.CommandError("Online migration expected to match one row when deleting '%s' in '%s'; %d found" % (version, self.context.version_table, ret.rowcount))