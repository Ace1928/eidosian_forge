from abc import ABCMeta, abstractmethod
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
class SubstringMatcher(BaseMatcher[str], metaclass=ABCMeta):

    def __init__(self, substring) -> None:
        if not isinstance(substring, str):
            raise TypeError(self.__class__.__name__ + ' requires string')
        self.substring = substring

    def describe_to(self, description: Description) -> None:
        description.append_text('a string ').append_text(self.relationship()).append_text(' ').append_description_of(self.substring)

    @abstractmethod
    def relationship(self):
        ...