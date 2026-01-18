import typing as t
class StructField(DataType):

    def __init__(self, name: str, dataType: DataType, nullable: bool=True, metadata: t.Optional[t.Dict[str, t.Any]]=None):
        self.name = name
        self.dataType = dataType
        self.nullable = nullable
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"StructField('{self.name}', {self.dataType}, {str(self.nullable)})"

    def simpleString(self) -> str:
        return f'{self.name}:{self.dataType.simpleString()}'

    def jsonValue(self) -> t.Dict[str, t.Any]:
        return {'name': self.name, 'type': self.dataType.jsonValue(), 'nullable': self.nullable, 'metadata': self.metadata}