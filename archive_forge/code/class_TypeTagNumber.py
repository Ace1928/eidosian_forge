import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
class TypeTagNumber(enum.IntEnum):
    end_of_content = 0
    boolean = 1
    integer = 2
    bit_string = 3
    octet_string = 4
    null = 5
    object_identifier = 6
    object_descriptor = 7
    external = 8
    real = 9
    enumerated = 10
    embedded_pdv = 11
    utf8_string = 12
    relative_oid = 13
    time = 14
    reserved = 15
    sequence = 16
    sequence_of = 16
    set = 17
    set_of = 17
    numeric_string = 18
    printable_string = 19
    t61_string = 20
    videotex_string = 21
    ia5_string = 22
    utc_time = 23
    generalized_time = 24
    graphic_string = 25
    visible_string = 26
    general_string = 27
    universal_string = 28
    character_string = 29
    bmp_string = 30
    date = 31
    time_of_day = 32
    date_time = 33
    duration = 34
    oid_iri = 35
    relative_oid_iri = 36

    @classmethod
    def native_labels(cls) -> typing.Dict[int, str]:
        return {TypeTagNumber.end_of_content: 'End-of-Content (EOC)', TypeTagNumber.boolean: 'BOOLEAN', TypeTagNumber.integer: 'INTEGER', TypeTagNumber.bit_string: 'BIT STRING', TypeTagNumber.octet_string: 'OCTET STRING', TypeTagNumber.null: 'NULL', TypeTagNumber.object_identifier: 'OBJECT IDENTIFIER', TypeTagNumber.object_descriptor: 'Object Descriptor', TypeTagNumber.external: 'EXTERNAL', TypeTagNumber.real: 'REAL (float)', TypeTagNumber.enumerated: 'ENUMERATED', TypeTagNumber.embedded_pdv: 'EMBEDDED PDV', TypeTagNumber.utf8_string: 'UTF8String', TypeTagNumber.relative_oid: 'RELATIVE-OID', TypeTagNumber.time: 'TIME', TypeTagNumber.reserved: 'RESERVED', TypeTagNumber.sequence: 'SEQUENCE or SEQUENCE OF', TypeTagNumber.set: 'SET or SET OF', TypeTagNumber.numeric_string: 'NumericString', TypeTagNumber.printable_string: 'PrintableString', TypeTagNumber.t61_string: 'T61String', TypeTagNumber.videotex_string: 'VideotexString', TypeTagNumber.ia5_string: 'IA5String', TypeTagNumber.utc_time: 'UTCTime', TypeTagNumber.generalized_time: 'GeneralizedTime', TypeTagNumber.graphic_string: 'GraphicString', TypeTagNumber.visible_string: 'VisibleString', TypeTagNumber.general_string: 'GeneralString', TypeTagNumber.universal_string: 'UniversalString', TypeTagNumber.character_string: 'CHARACTER', TypeTagNumber.bmp_string: 'BMPString', TypeTagNumber.date: 'DATE', TypeTagNumber.time_of_day: 'TIME-OF-DAY', TypeTagNumber.date_time: 'DATE-TIME', TypeTagNumber.duration: 'DURATION', TypeTagNumber.oid_iri: 'OID-IRI', TypeTagNumber.relative_oid_iri: 'RELATIVE-OID-IRI'}