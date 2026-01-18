import argparse
from textwrap import indent
import xml.etree.ElementTree as ET
from jeepney.wrappers import Introspectable
from jeepney.io.blocking import open_dbus_connection, Proxy
from jeepney import __version__
from jeepney.wrappers import MessageGenerator, new_method_call
def _make_code_noargs(self):
    return "def {name}(self):\n    return new_method_call(self, '{name}')\n".format(name=self.name)