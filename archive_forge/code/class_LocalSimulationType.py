import enum
class LocalSimulationType(enum.Enum):
    SYNCHRONOUS = 1
    ASYNCHRONOUS = 2
    ASYNCHRONOUS_WITH_DELAY = 3