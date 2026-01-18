import enum
class ActionStatus(str, enum.Enum):
    done = 'done'
    error = 'error'
    queued = 'queued'
    executing = 'executing'
    unknown = 'unknown'

    def __str__(self):
        return self.value