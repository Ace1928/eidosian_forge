from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
def _validate_TCP_UDP_SCTP_port_range(port_number, parameter_name):
    if port_number < constants.MIN_PORT_NUMBER or port_number > constants.MAX_PORT_NUMBER:
        msg = "Invalid input for field/attribute '{name}', Value: '{port}'. Value must be between {pmin} and {pmax}.".format(name=parameter_name, port=port_number, pmin=constants.MIN_PORT_NUMBER, pmax=constants.MAX_PORT_NUMBER)
        raise exceptions.InvalidValue(msg)