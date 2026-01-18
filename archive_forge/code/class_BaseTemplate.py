from magnumclient.common import base
from magnumclient.common import utils
from magnumclient import exceptions
class BaseTemplate(base.Resource):
    template_name = ''

    def __repr__(self):
        return '<' + self.__class__.template_name + ' %s>' % self._info