import functools
from pydantic import BaseModel, computed_field
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
class SQLiteModelConfig(BaseModel):
    """
    The SQLite Model Schema
    """
    tablename: str
    sql_fields: Dict[str, Union[str, List[str], Dict[str, str]]]
    sql_pkey: str
    sql_keys: List[str]
    sql_insert: str
    sql_insert_q: str
    search_precisions: Dict[str, int]
    autoset: Optional[bool] = None
    sql_model_name: Optional[str] = None
    db_conn: Optional[Any] = None
    if TYPE_CHECKING:
        db_conn: Union['sqlite3.Connection', 'aiosqlite.Connection'] = None

    @property
    def conn(self) -> Union['sqlite3.Connection', 'aiosqlite.Connection']:
        """
        Returns the connection
        """
        if self.db_conn is None:
            if self.sql_model_name in _sqlite_model_name_to_connection:
                self.db_conn = _sqlite_model_name_to_connection[self.sql_model_name]
            else:
                raise ValueError(f'Model {self.sql_model_name} not registered and no connection provided')
        return self.db_conn

    def __getitem__(self, key: str) -> Any:
        """
        Gets the item
        """
        return getattr(self, key)

    @computed_field
    @property
    def sql_schema(self) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
        """
        Returns the SQLite Schema
        """
        return {'tablename': self.tablename, 'sql_fields': self.sql_fields, 'sql_pkey': self.sql_pkey, 'sql_keys': self.sql_keys, 'sql_insert': self.sql_insert, 'sql_insert_q': self.sql_insert_q, 'search_precisions': self.search_precisions, 'autoset': self.autoset}