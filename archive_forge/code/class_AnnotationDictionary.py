from abc import ABC
from ..constants import AnnotationFlag
from ..generic import NameObject, NumberObject
from ..generic._data_structures import DictionaryObject
class AnnotationDictionary(DictionaryObject, ABC):

    def __init__(self) -> None:
        from ..generic._base import NameObject
        self[NameObject('/Type')] = NameObject('/Annot')

    @property
    def flags(self) -> AnnotationFlag:
        return self.get(NameObject('/F'), AnnotationFlag(0))

    @flags.setter
    def flags(self, value: AnnotationFlag) -> None:
        self[NameObject('/F')] = NumberObject(value)