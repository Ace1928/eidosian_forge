import logging
import time
def _enter_create(self, value, createdtime):
    if not self._is_expired(createdtime):
        return NOT_REGENERATED
    _async = False
    if self._has_value(createdtime):
        has_value = True
        if not self.mutex.acquire(False):
            log.debug('creation function in progress elsewhere, returning')
            return NOT_REGENERATED
    else:
        has_value = False
        log.debug('no value, waiting for create lock')
        self.mutex.acquire()
    try:
        log.debug('value creation lock %r acquired' % self.mutex)
        if not has_value:
            try:
                value, createdtime = self.value_and_created_fn()
            except NeedRegenerationException:
                pass
            else:
                has_value = True
                if not self._is_expired(createdtime):
                    log.debug('Concurrent thread created the value')
                    return (value, createdtime)
        if has_value and self.async_creator:
            log.debug('Passing creation lock to async runner')
            self.async_creator(self.mutex)
            _async = True
            return (value, createdtime)
        log.debug('Calling creation function for %s value', 'not-yet-present' if not has_value else 'previously expired')
        return self.creator()
    finally:
        if not _async:
            self.mutex.release()
            log.debug('Released creation lock')