from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
class BoundModelBase:
    """Bound Model Base"""
    model: Any

    def __init__(self, client: ClientEntityBase, data: dict, complete: bool=True):
        """
        :param client:
                The client for the specific model to use
        :param data:
                The data of the model
        :param complete: bool
                False if not all attributes of the model fetched
        """
        self._client = client
        self.complete = complete
        self.data_model = self.model.from_dict(data)

    def __getattr__(self, name: str):
        """Allow magical access to the properties of the model
        :param name: str
        :return:
        """
        value = getattr(self.data_model, name)
        if not value and (not self.complete):
            self.reload()
            value = getattr(self.data_model, name)
        return value

    def reload(self) -> None:
        """Reloads the model and tries to get all data from the APIx"""
        assert hasattr(self._client, 'get_by_id')
        bound_model = self._client.get_by_id(self.data_model.id)
        self.data_model = bound_model.data_model
        self.complete = True

    def __repr__(self) -> str:
        return object.__repr__(self)