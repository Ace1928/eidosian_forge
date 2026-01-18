import json
from typing import List
import jinja2
from minerl.herobraine.hero.mc import EQUIPMENT_SLOTS
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import TranslationHandler, TranslationHandlerGroup
import numpy as np
class _TypeObservation(TranslationHandler):
    """
    Returns the item list index  of the tool in the given hand
    List must start with 'none' as 0th element and end with 'other' as wildcard element
    # TODO (R): Update this dcoumentation
    """

    def __init__(self, keys: List[str], items: list, _default: str, _other: str):
        """
        Initializes the space of the handler with a spaces.Dict
        of all of the spaces for each individual command.
        """
        self._items = sorted(items)
        self._keys = keys
        self._univ_items = ['minecraft:' + item for item in items]
        self._default = _default
        self._other = _other
        if _other not in self._items or _default not in self._items:
            print(self._items)
            print(_default)
            print(_other)
        assert self._other in items
        assert self._default in items
        super().__init__(spaces.Enum(*self._items, default=self._default))

    def to_string(self):
        return 'type'

    def from_hero(self, obs_dict):
        try:
            head = obs_dict['equipped_items']
            for key in self._keys:
                head = json.loads(head[key])
            item = head['type']
            return self._other if item not in self._items else item
        except KeyError:
            return self._default

    def from_universal(self, obs):
        try:
            if self._keys[0] == 'mainhand' and len(self._keys) == 1:
                offset = -9
                hotbar_index = obs['hotbar']
                if obs['slots']['gui']['type'] == 'class net.minecraft.inventory.ContainerPlayer':
                    offset -= 1
                item_name = obs['slots']['gui']['slots'][offset + hotbar_index]['name'].split('minecraft:')[-1]
                if not item_name in self._items:
                    raise ValueError()
                if item_name == 'air':
                    raise KeyError()
                return item_name
            else:
                raise NotImplementedError('type not implemented for hand type' + str(self._keys))
        except KeyError:
            return self._default
        except ValueError:
            return self._other

    def __or__(self, other):
        """
        Combines two TypeObservation's (self and other) into one by 
        taking the union of self.items and other.items
        """
        if isinstance(other, _TypeObservation):
            return _TypeObservation(self._keys, list(set(self._items + other._items)), _other=self._other, _default=self._default)
        else:
            raise TypeError('Operands have to be of type TypeObservation')

    def __eq__(self, other):
        return self._keys == other._keys and self._items == other._items