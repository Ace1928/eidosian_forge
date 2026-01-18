import asyncio
import pytest
import pyarrow
class ExampleServer(flight.FlightServerBase):
    simple_info = flight.FlightInfo(pyarrow.schema([('a', 'int32')]), flight.FlightDescriptor.for_command(b'simple'), [], -1, -1)

    def get_flight_info(self, context, descriptor):
        if descriptor.command == b'simple':
            return self.simple_info
        elif descriptor.command == b'unknown':
            raise NotImplementedError('Unknown command')
        raise NotImplementedError('Unknown descriptor')