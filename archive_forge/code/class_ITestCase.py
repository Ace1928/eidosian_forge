import zope.interface as zi
class ITestCase(zi.Interface):
    """
    The interface that a test case must implement in order to be used in Trial.
    """
    failureException = zi.Attribute('The exception class that is raised by failed assertions')

    def __call__(result):
        """
        Run the test. Should always do exactly the same thing as run().
        """

    def countTestCases():
        """
        Return the number of tests in this test case. Usually 1.
        """

    def id():
        """
        Return a unique identifier for the test, usually the fully-qualified
        Python name.
        """

    def run(result):
        """
        Run the test, storing the results in C{result}.

        @param result: A L{TestResult}.
        """

    def shortDescription():
        """
        Return a short description of the test.
        """