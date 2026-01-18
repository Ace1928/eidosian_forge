import contextlib
import decimal
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence, Union
from sqlalchemy import (
from sqlalchemy.ext.asyncio import (
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query, Session, sessionmaker
from langchain.indexes.base import RecordManager
Delete records from the SQLite database.