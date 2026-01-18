from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
class IsEqualIgnoringWhiteSpace(BaseMatcher[str]):

    def __init__(self, string) -> None:
        if not isinstance(string, str):
            raise TypeError('IsEqualIgnoringWhiteSpace requires string')
        self.original_string = string
        self.stripped_string = stripspace(string)

    def _matches(self, item: str) -> bool:
        if not isinstance(item, str):
            return False
        return self.stripped_string == stripspace(item)

    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.original_string).append_text(' ignoring whitespace')