from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
class UnaccentExtension(CreateExtension):

    def __init__(self):
        self.name = 'unaccent'