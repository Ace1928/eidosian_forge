from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, ListMixin, RefreshMixin, RetrieveMixin
from gitlab.types import RequiredOptional
class BulkImport(RefreshMixin, RESTObject):
    entities: 'BulkImportEntityManager'