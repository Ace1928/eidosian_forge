from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from mlflow.store.tracking.dbmodels.models import SqlDataset, SqlInputTag, SqlInput
add datasets inputs input_tags tables

Revision ID: 7f2a7d5fae7d
Revises: 3500859a5d39
Create Date: 2023-03-23 09:48:27.775166

