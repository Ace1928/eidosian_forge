from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.matcher import Matcher
from hamcrest.library.text.substringmatcher import SubstringMatcher
class StringStartsWith(SubstringMatcher):

    def __init__(self, substring) -> None:
        super(StringStartsWith, self).__init__(substring)

    def _matches(self, item: str) -> bool:
        if not hasmethod(item, 'startswith'):
            return False
        return item.startswith(self.substring)

    def relationship(self):
        return 'starting with'