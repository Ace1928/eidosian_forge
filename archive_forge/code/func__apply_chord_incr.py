from kombu.utils.encoding import bytes_to_str, ensure_bytes
from kombu.utils.objects import cached_property
from celery.exceptions import ImproperlyConfigured
from celery.utils.functional import LRUCache
from .base import KeyValueStoreBackend
def _apply_chord_incr(self, header_result_args, body, **kwargs):
    chord_key = self.get_key_for_chord(header_result_args[0])
    self.client.set(chord_key, 0, time=self.expires)
    return super()._apply_chord_incr(header_result_args, body, **kwargs)