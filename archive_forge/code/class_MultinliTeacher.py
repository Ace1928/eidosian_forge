from parlai.core.teachers import MultiTaskTeacher
import parlai.tasks.anli.agents as anli
import parlai.tasks.multinli.agents as multinli
import parlai.tasks.snli.agents as snli
import parlai.tasks.dialogue_nli.agents as dnli
from copy import deepcopy
class MultinliTeacher(multinli.DefaultTeacher):
    pass