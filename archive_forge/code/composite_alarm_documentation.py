from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import alarm_base
from heat.engine import support
A resource that implements Aodh composite alarm.

    Allows to specify multiple rules when creating a composite alarm,
    and the rules combined with logical operators: and, or.
    