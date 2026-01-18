from pathlib import Path
from alembic.command import upgrade
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine.base import Engine
def _get_alembic_dir() -> str:
    return Path(__file__).parent / 'migrations'