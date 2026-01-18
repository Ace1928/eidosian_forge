import dataclasses
import socket
import ssl
import threading
import typing as t
def _read_loop(self) -> None:
    data_buffer = bytearray()
    while True:
        try:
            resp = self._sock.recv(4096)
            if not resp:
                raise Exception('LDAP connection has been shutdown')
            data_buffer.extend(resp)
            while data_buffer:
                if self._encryptor:
                    dec_data, enc_len = self._encryptor.unwrap(data_buffer)
                    if enc_len == 0:
                        break
                    data_buffer = data_buffer[enc_len:]
                else:
                    dec_data = bytes(data_buffer)
                    data_buffer = bytearray()
                for msg in self._protocol.receive(dec_data):
                    for handler in self._response_handler:
                        handler.append(msg)
        except sansldap.ProtocolError as e:
            if e.response:
                self._sock.sendall(e.response)
            for handler in self._response_handler:
                handler.append(e)
            break
        except Exception as e:
            for handler in self._response_handler:
                handler.append(e)
            break