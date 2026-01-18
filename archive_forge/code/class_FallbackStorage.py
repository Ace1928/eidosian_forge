from django.contrib.messages.storage.base import BaseStorage
from django.contrib.messages.storage.cookie import CookieStorage
from django.contrib.messages.storage.session import SessionStorage
class FallbackStorage(BaseStorage):
    """
    Try to store all messages in the first backend. Store any unstored
    messages in each subsequent backend.
    """
    storage_classes = (CookieStorage, SessionStorage)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storages = [storage_class(*args, **kwargs) for storage_class in self.storage_classes]
        self._used_storages = set()

    def _get(self, *args, **kwargs):
        """
        Get a single list of messages from all storage backends.
        """
        all_messages = []
        for storage in self.storages:
            messages, all_retrieved = storage._get()
            if messages is None:
                break
            if messages:
                self._used_storages.add(storage)
            all_messages.extend(messages)
            if all_retrieved:
                break
        return (all_messages, all_retrieved)

    def _store(self, messages, response, *args, **kwargs):
        """
        Store the messages and return any unstored messages after trying all
        backends.

        For each storage backend, any messages not stored are passed on to the
        next backend.
        """
        for storage in self.storages:
            if messages:
                messages = storage._store(messages, response, remove_oldest=False)
            elif storage in self._used_storages:
                storage._store([], response)
                self._used_storages.remove(storage)
        return messages