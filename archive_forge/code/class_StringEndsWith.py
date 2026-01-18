from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.matcher import Matcher
from hamcrest.library.text.substringmatcher import SubstringMatcher
class StringEndsWith(SubstringMatcher):

    def __init__(self, substring) -> None:
        super(StringEndsWith, self).__init__(substring)

    def _matches(self, item: str) -> bool:
        if not hasmethod(item, 'endswith'):
            return False
        return item.endswith(self.substring)

    def relationship(self):
        return 'ending with'