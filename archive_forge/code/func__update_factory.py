from modin.config import Engine, IsExperimental, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.utils import _inherit_docstrings, get_current_execution
@classmethod
def _update_factory(cls, *args):
    """
        Update and prepare factory with a new one specified via Modin config.

        Parameters
        ----------
        *args : iterable
            This parameters serves the compatibility purpose.
            Does not affect the result.
        """
    factory_name = get_current_execution() + 'Factory'
    experimental_factory_name = 'Experimental' + factory_name
    try:
        cls.__factory = getattr(factories, factory_name, None) or getattr(factories, experimental_factory_name)
    except AttributeError:
        if not IsExperimental.get():
            msg = 'Cannot find neither factory {} nor experimental factory {}. ' + 'Potential reason might be incorrect environment variable value for ' + f'{StorageFormat.varname} or {Engine.varname}'
            raise FactoryNotFoundError(msg.format(factory_name, experimental_factory_name))
        cls.__factory = StubFactory.set_failing_name(factory_name)
    else:
        try:
            cls.__factory.prepare()
        except ModuleNotFoundError as err:
            cls.__factory = None
            raise ModuleNotFoundError(f'Make sure all required packages are installed: {str(err)}') from err
        except BaseException:
            cls.__factory = None
            raise