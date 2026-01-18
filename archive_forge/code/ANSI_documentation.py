from . import screen
from . import FSM
import string
Handler for [?<number>h and [?<number>l. If anyone
        wanted to actually use these, they'd need to add more states to the
        FSM rather than just improve or override this method. 