from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
from argparse import ArgumentParser
from .._utils import get_client, print_model
from .._models import BaseModel
from .._progress import BufferReader
class CLIFile:

    @staticmethod
    def create(args: CLIFileCreateArgs) -> None:
        with open(args.file, 'rb') as file_reader:
            buffer_reader = BufferReader(file_reader.read(), desc='Upload progress')
        file = get_client().files.create(file=(args.file, buffer_reader), purpose=cast(Any, args.purpose))
        print_model(file)

    @staticmethod
    def get(args: CLIFileIDArgs) -> None:
        file = get_client().files.retrieve(file_id=args.id)
        print_model(file)

    @staticmethod
    def delete(args: CLIFileIDArgs) -> None:
        file = get_client().files.delete(file_id=args.id)
        print_model(file)

    @staticmethod
    def list() -> None:
        files = get_client().files.list()
        for file in files:
            print_model(file)