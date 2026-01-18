import enum
import gast
class Static(NoValue):
    """Container for static analysis annotation keys.

  The enum values are used strictly for documentation purposes.
  """
    IS_PARAM = 'Symbol is a parameter to the function being analyzed.'
    SCOPE = 'The scope for the annotated node. See activity.py.'
    ARGS_SCOPE = 'The scope for the argument list of a function call.'
    COND_SCOPE = 'The scope for the test node of a conditional statement.'
    BODY_SCOPE = 'The scope for the main body of a statement (True branch for if statements, main body for loops).'
    ORELSE_SCOPE = 'The scope for the orelse body of a statement (False branch for if statements, orelse body for loops).'
    DEFINITIONS = 'Reaching definition information. See reaching_definitions.py.'
    ORIG_DEFINITIONS = 'The value of DEFINITIONS that applied to the original code before any conversion.'
    DEFINED_FNS_IN = 'Local function definitions that may exist when exiting the node. See reaching_fndefs.py'
    DEFINED_VARS_IN = 'Symbols defined when entering the node. See reaching_definitions.py.'
    LIVE_VARS_OUT = 'Symbols live when exiting the node. See liveness.py.'
    LIVE_VARS_IN = 'Symbols live when entering the node. See liveness.py.'
    TYPES = 'Static type information. See type_inference.py.'
    CLOSURE_TYPES = 'Types of closure symbols at each detected call site.'
    VALUE = 'Static value information. See type_inference.py.'