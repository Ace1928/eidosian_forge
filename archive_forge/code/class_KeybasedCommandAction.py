from minerl.herobraine.hero.handlers.translation import TranslationHandler
import typing
from minerl.herobraine.hero.handlers.agent.action import Action, ItemListAction
import jinja2
import minerl.herobraine.hero.spaces as spaces
class KeybasedCommandAction(Action):
    """
    A command action which is generated from human keypresses in anvil.
    Examples of such actions are movement actions, etc.

    This is not to be confused with keyboard acitons, wehreby both anvil and malmo
    simulate and act on direct key codes.

    Combinations of KeybasedCommandActions yield acitons like:
    {
			“move” : 1,
			“jump”: 1 
    } 
    where move and jump are hte commands, which correspond to keys like 'W', 'SPACE', etc.

    This is as opposed to keyboard actions (see the following class definition in keyboard.py)
    which yield actions like:
    {
        "keyboard" : {
            "W" : 1,
            "A": 1,
            "S": 0,
            "E": 1,
            ...
        }
    }
    More information can be found in the unification document (internal).
    """

    def to_string(self):
        return self.command

    def xml_template(self) -> str:
        """Notice how all of the instances of keybased command actions,
        of which there will be typically many in an environment spec,
        correspond to exactly the same XML stub.

        This is discussed at length in the unification proposal
        and is a chief example of where manifest consolidation is needed.
        """
        return str('<HumanLevelCommands/>')

    def __init__(self, command, *keys):
        if len(keys) == 2:
            super().__init__(command, spaces.DiscreteRange(-1, 2))
        else:
            super().__init__(command, spaces.Discrete(len(keys) + 1))
        self.keys = keys

    def from_universal(self, x):
        actions_mapped = set((str(k) for k in x['custom_action']['actions']['keys']))
        offset = self.space.begin if isinstance(self.space, spaces.DiscreteRange) else 0
        default = 0
        for i, key in enumerate(self.keys):
            if key in actions_mapped:
                if isinstance(self.space, spaces.DiscreteRange):
                    return i * 2 + offset
                else:
                    return i + 1 + offset
        return default