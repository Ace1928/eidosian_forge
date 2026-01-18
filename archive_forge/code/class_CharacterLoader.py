from asyncio import gather
from collections import namedtuple
from functools import partial
from unittest.mock import Mock
from graphene.utils.dataloader import DataLoader
from pytest import mark, raises
from graphene import ObjectType, String, Schema, Field, List
class CharacterLoader(DataLoader):

    async def batch_load_fn(self, character_ids):
        return mock_batch_load_fn(character_ids)