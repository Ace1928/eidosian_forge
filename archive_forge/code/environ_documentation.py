import os
from fixtures import Fixture
Create an EnvironmentVariable fixture.

        :param varname: the name of the variable to isolate.
        :param newvalue: A value to set the variable to. If None, the variable
            will be deleted.

        During setup the variable will be deleted or assigned the requested
        value, and this will be restored in cleanUp.
        