import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import pandas as pd
import pyarrow as pa
import datasets
import datasets.config
from datasets.features.features import require_storage_cast
from datasets.table import table_cast
@dataclass
class SqlConfig(datasets.BuilderConfig):
    """BuilderConfig for SQL."""
    sql: Union[str, 'sqlalchemy.sql.Selectable'] = None
    con: Union[str, 'sqlalchemy.engine.Connection', 'sqlalchemy.engine.Engine', 'sqlite3.Connection'] = None
    index_col: Optional[Union[str, List[str]]] = None
    coerce_float: bool = True
    params: Optional[Union[List, Tuple, Dict]] = None
    parse_dates: Optional[Union[List, Dict]] = None
    columns: Optional[List[str]] = None
    chunksize: Optional[int] = 10000
    features: Optional[datasets.Features] = None

    def __post_init__(self):
        if self.sql is None:
            raise ValueError('sql must be specified')
        if self.con is None:
            raise ValueError('con must be specified')

    def create_config_id(self, config_kwargs: dict, custom_features: Optional[datasets.Features]=None) -> str:
        config_kwargs = config_kwargs.copy()
        sql = config_kwargs['sql']
        if not isinstance(sql, str):
            if datasets.config.SQLALCHEMY_AVAILABLE and 'sqlalchemy' in sys.modules:
                import sqlalchemy
                if isinstance(sql, sqlalchemy.sql.Selectable):
                    engine = sqlalchemy.create_engine(config_kwargs['con'].split('://')[0] + '://')
                    sql_str = str(sql.compile(dialect=engine.dialect))
                    config_kwargs['sql'] = sql_str
                else:
                    raise TypeError(f"Supported types for 'sql' are string and sqlalchemy.sql.Selectable but got {type(sql)}: {sql}")
            else:
                raise TypeError(f"Supported types for 'sql' are string and sqlalchemy.sql.Selectable but got {type(sql)}: {sql}")
        con = config_kwargs['con']
        if not isinstance(con, str):
            config_kwargs['con'] = id(con)
            logger.info(f"SQL connection 'con' of type {type(con)} couldn't be hashed properly. To enable hashing, specify 'con' as URI string instead.")
        return super().create_config_id(config_kwargs, custom_features=custom_features)

    @property
    def pd_read_sql_kwargs(self):
        pd_read_sql_kwargs = {'index_col': self.index_col, 'columns': self.columns, 'params': self.params, 'coerce_float': self.coerce_float, 'parse_dates': self.parse_dates}
        return pd_read_sql_kwargs