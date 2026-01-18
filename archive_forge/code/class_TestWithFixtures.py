import unittest
from fixtures.fixture import gather_details
class TestWithFixtures(unittest.TestCase):
    """A TestCase with a helper function to use fixtures.

    Normally used as a mix-in class to add useFixture.

    Note that test classes such as testtools.TestCase which already have a
    ``useFixture`` method do not need this mixed in.
    """

    def useFixture(self, fixture):
        """Use fixture in a test case.

        The fixture will be setUp, and self.addCleanup(fixture.cleanUp) called.

        :param fixture: The fixture to use.
        :return: The fixture, after setting it up and scheduling a cleanup for
           it.
        """
        use_details = gather_details is not None and getattr(self, 'addDetail', None) is not None
        try:
            fixture.setUp()
        except:
            if use_details:
                gather_details(fixture.getDetails(), self.getDetails())
            raise
        else:
            self.addCleanup(fixture.cleanUp)
            if use_details:
                self.addCleanup(gather_details, fixture, self)
            return fixture