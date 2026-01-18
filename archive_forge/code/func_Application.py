from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer
from twisted.persisted import sob
from twisted.plugin import IPlugin
from twisted.python import components
from twisted.python.reflect import namedAny
def Application(name, uid=None, gid=None):
    """
    Return a compound class.

    Return an object supporting the L{IService}, L{IServiceCollection},
    L{IProcess} and L{sob.IPersistable} interfaces, with the given
    parameters. Always access the return value by explicit casting to
    one of the interfaces.
    """
    ret = components.Componentized()
    availableComponents = [MultiService(), Process(uid, gid), sob.Persistent(ret, name)]
    for comp in availableComponents:
        ret.addComponent(comp, ignoreClass=1)
    IService(ret).setName(name)
    return ret