from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm, Column, Integer, String, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.orm import relationship, backref, declarative_base
from mlflow.utils.mlflow_tags import MLFLOW_USER
migrate user column to tags

Revision ID: 90e64c465722
Revises: 451aebb31d03
Create Date: 2019-05-29 10:43:52.919427

