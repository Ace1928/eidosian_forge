import itertools
import sys
from fixtures.callmany import (
class CompoundFixture(Fixture):
    """A fixture that combines many fixtures.

    :ivar fixtures: The list of fixtures that make up this one. (read only).
    """

    def __init__(self, fixtures):
        """Construct a fixture made of many fixtures.

        :param fixtures: An iterable of fixtures.
        """
        super(CompoundFixture, self).__init__()
        self.fixtures = list(fixtures)

    def _setUp(self):
        for fixture in self.fixtures:
            self.useFixture(fixture)