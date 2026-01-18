import enum
@enum.unique
class SaveType(str, enum.Enum):
    SAVEDMODEL = 'savedmodel'
    CHECKPOINT = 'checkpoint'