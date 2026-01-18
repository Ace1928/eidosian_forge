from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class PropertyValue(ProtocolBuffer.ProtocolMessage):
    has_int64value_ = 0
    int64value_ = 0
    has_booleanvalue_ = 0
    booleanvalue_ = 0
    has_stringvalue_ = 0
    stringvalue_ = ''
    has_doublevalue_ = 0
    doublevalue_ = 0.0
    has_pointvalue_ = 0
    pointvalue_ = None
    has_uservalue_ = 0
    uservalue_ = None
    has_referencevalue_ = 0
    referencevalue_ = None

    def __init__(self, contents=None):
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def int64value(self):
        return self.int64value_

    def set_int64value(self, x):
        self.has_int64value_ = 1
        self.int64value_ = x

    def clear_int64value(self):
        if self.has_int64value_:
            self.has_int64value_ = 0
            self.int64value_ = 0

    def has_int64value(self):
        return self.has_int64value_

    def booleanvalue(self):
        return self.booleanvalue_

    def set_booleanvalue(self, x):
        self.has_booleanvalue_ = 1
        self.booleanvalue_ = x

    def clear_booleanvalue(self):
        if self.has_booleanvalue_:
            self.has_booleanvalue_ = 0
            self.booleanvalue_ = 0

    def has_booleanvalue(self):
        return self.has_booleanvalue_

    def stringvalue(self):
        return self.stringvalue_

    def set_stringvalue(self, x):
        self.has_stringvalue_ = 1
        self.stringvalue_ = x

    def clear_stringvalue(self):
        if self.has_stringvalue_:
            self.has_stringvalue_ = 0
            self.stringvalue_ = ''

    def has_stringvalue(self):
        return self.has_stringvalue_

    def doublevalue(self):
        return self.doublevalue_

    def set_doublevalue(self, x):
        self.has_doublevalue_ = 1
        self.doublevalue_ = x

    def clear_doublevalue(self):
        if self.has_doublevalue_:
            self.has_doublevalue_ = 0
            self.doublevalue_ = 0.0

    def has_doublevalue(self):
        return self.has_doublevalue_

    def pointvalue(self):
        if self.pointvalue_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.pointvalue_ is None:
                    self.pointvalue_ = PropertyValue_PointValue()
            finally:
                self.lazy_init_lock_.release()
        return self.pointvalue_

    def mutable_pointvalue(self):
        self.has_pointvalue_ = 1
        return self.pointvalue()

    def clear_pointvalue(self):
        if self.has_pointvalue_:
            self.has_pointvalue_ = 0
            if self.pointvalue_ is not None:
                self.pointvalue_.Clear()

    def has_pointvalue(self):
        return self.has_pointvalue_

    def uservalue(self):
        if self.uservalue_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.uservalue_ is None:
                    self.uservalue_ = PropertyValue_UserValue()
            finally:
                self.lazy_init_lock_.release()
        return self.uservalue_

    def mutable_uservalue(self):
        self.has_uservalue_ = 1
        return self.uservalue()

    def clear_uservalue(self):
        if self.has_uservalue_:
            self.has_uservalue_ = 0
            if self.uservalue_ is not None:
                self.uservalue_.Clear()

    def has_uservalue(self):
        return self.has_uservalue_

    def referencevalue(self):
        if self.referencevalue_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.referencevalue_ is None:
                    self.referencevalue_ = PropertyValue_ReferenceValue()
            finally:
                self.lazy_init_lock_.release()
        return self.referencevalue_

    def mutable_referencevalue(self):
        self.has_referencevalue_ = 1
        return self.referencevalue()

    def clear_referencevalue(self):
        if self.has_referencevalue_:
            self.has_referencevalue_ = 0
            if self.referencevalue_ is not None:
                self.referencevalue_.Clear()

    def has_referencevalue(self):
        return self.has_referencevalue_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_int64value():
            self.set_int64value(x.int64value())
        if x.has_booleanvalue():
            self.set_booleanvalue(x.booleanvalue())
        if x.has_stringvalue():
            self.set_stringvalue(x.stringvalue())
        if x.has_doublevalue():
            self.set_doublevalue(x.doublevalue())
        if x.has_pointvalue():
            self.mutable_pointvalue().MergeFrom(x.pointvalue())
        if x.has_uservalue():
            self.mutable_uservalue().MergeFrom(x.uservalue())
        if x.has_referencevalue():
            self.mutable_referencevalue().MergeFrom(x.referencevalue())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_int64value_ != x.has_int64value_:
            return 0
        if self.has_int64value_ and self.int64value_ != x.int64value_:
            return 0
        if self.has_booleanvalue_ != x.has_booleanvalue_:
            return 0
        if self.has_booleanvalue_ and self.booleanvalue_ != x.booleanvalue_:
            return 0
        if self.has_stringvalue_ != x.has_stringvalue_:
            return 0
        if self.has_stringvalue_ and self.stringvalue_ != x.stringvalue_:
            return 0
        if self.has_doublevalue_ != x.has_doublevalue_:
            return 0
        if self.has_doublevalue_ and self.doublevalue_ != x.doublevalue_:
            return 0
        if self.has_pointvalue_ != x.has_pointvalue_:
            return 0
        if self.has_pointvalue_ and self.pointvalue_ != x.pointvalue_:
            return 0
        if self.has_uservalue_ != x.has_uservalue_:
            return 0
        if self.has_uservalue_ and self.uservalue_ != x.uservalue_:
            return 0
        if self.has_referencevalue_ != x.has_referencevalue_:
            return 0
        if self.has_referencevalue_ and self.referencevalue_ != x.referencevalue_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if self.has_pointvalue_ and (not self.pointvalue_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_uservalue_ and (not self.uservalue_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_referencevalue_ and (not self.referencevalue_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_int64value_:
            n += 1 + self.lengthVarInt64(self.int64value_)
        if self.has_booleanvalue_:
            n += 2
        if self.has_stringvalue_:
            n += 1 + self.lengthString(len(self.stringvalue_))
        if self.has_doublevalue_:
            n += 9
        if self.has_pointvalue_:
            n += 2 + self.pointvalue_.ByteSize()
        if self.has_uservalue_:
            n += 2 + self.uservalue_.ByteSize()
        if self.has_referencevalue_:
            n += 2 + self.referencevalue_.ByteSize()
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_int64value_:
            n += 1 + self.lengthVarInt64(self.int64value_)
        if self.has_booleanvalue_:
            n += 2
        if self.has_stringvalue_:
            n += 1 + self.lengthString(len(self.stringvalue_))
        if self.has_doublevalue_:
            n += 9
        if self.has_pointvalue_:
            n += 2 + self.pointvalue_.ByteSizePartial()
        if self.has_uservalue_:
            n += 2 + self.uservalue_.ByteSizePartial()
        if self.has_referencevalue_:
            n += 2 + self.referencevalue_.ByteSizePartial()
        return n

    def Clear(self):
        self.clear_int64value()
        self.clear_booleanvalue()
        self.clear_stringvalue()
        self.clear_doublevalue()
        self.clear_pointvalue()
        self.clear_uservalue()
        self.clear_referencevalue()

    def OutputUnchecked(self, out):
        if self.has_int64value_:
            out.putVarInt32(8)
            out.putVarInt64(self.int64value_)
        if self.has_booleanvalue_:
            out.putVarInt32(16)
            out.putBoolean(self.booleanvalue_)
        if self.has_stringvalue_:
            out.putVarInt32(26)
            out.putPrefixedString(self.stringvalue_)
        if self.has_doublevalue_:
            out.putVarInt32(33)
            out.putDouble(self.doublevalue_)
        if self.has_pointvalue_:
            out.putVarInt32(43)
            self.pointvalue_.OutputUnchecked(out)
            out.putVarInt32(44)
        if self.has_uservalue_:
            out.putVarInt32(67)
            self.uservalue_.OutputUnchecked(out)
            out.putVarInt32(68)
        if self.has_referencevalue_:
            out.putVarInt32(99)
            self.referencevalue_.OutputUnchecked(out)
            out.putVarInt32(100)

    def OutputPartial(self, out):
        if self.has_int64value_:
            out.putVarInt32(8)
            out.putVarInt64(self.int64value_)
        if self.has_booleanvalue_:
            out.putVarInt32(16)
            out.putBoolean(self.booleanvalue_)
        if self.has_stringvalue_:
            out.putVarInt32(26)
            out.putPrefixedString(self.stringvalue_)
        if self.has_doublevalue_:
            out.putVarInt32(33)
            out.putDouble(self.doublevalue_)
        if self.has_pointvalue_:
            out.putVarInt32(43)
            self.pointvalue_.OutputPartial(out)
            out.putVarInt32(44)
        if self.has_uservalue_:
            out.putVarInt32(67)
            self.uservalue_.OutputPartial(out)
            out.putVarInt32(68)
        if self.has_referencevalue_:
            out.putVarInt32(99)
            self.referencevalue_.OutputPartial(out)
            out.putVarInt32(100)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_int64value(d.getVarInt64())
                continue
            if tt == 16:
                self.set_booleanvalue(d.getBoolean())
                continue
            if tt == 26:
                self.set_stringvalue(d.getPrefixedString())
                continue
            if tt == 33:
                self.set_doublevalue(d.getDouble())
                continue
            if tt == 43:
                self.mutable_pointvalue().TryMerge(d)
                continue
            if tt == 67:
                self.mutable_uservalue().TryMerge(d)
                continue
            if tt == 99:
                self.mutable_referencevalue().TryMerge(d)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_int64value_:
            res += prefix + 'int64Value: %s\n' % self.DebugFormatInt64(self.int64value_)
        if self.has_booleanvalue_:
            res += prefix + 'booleanValue: %s\n' % self.DebugFormatBool(self.booleanvalue_)
        if self.has_stringvalue_:
            res += prefix + 'stringValue: %s\n' % self.DebugFormatString(self.stringvalue_)
        if self.has_doublevalue_:
            res += prefix + 'doubleValue: %s\n' % self.DebugFormat(self.doublevalue_)
        if self.has_pointvalue_:
            res += prefix + 'PointValue {\n'
            res += self.pointvalue_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
        if self.has_uservalue_:
            res += prefix + 'UserValue {\n'
            res += self.uservalue_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
        if self.has_referencevalue_:
            res += prefix + 'ReferenceValue {\n'
            res += self.referencevalue_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kint64Value = 1
    kbooleanValue = 2
    kstringValue = 3
    kdoubleValue = 4
    kPointValueGroup = 5
    kPointValuex = 6
    kPointValuey = 7
    kUserValueGroup = 8
    kUserValueemail = 9
    kUserValueauth_domain = 10
    kUserValuenickname = 11
    kUserValuegaiaid = 18
    kUserValueobfuscated_gaiaid = 19
    kUserValuefederated_identity = 21
    kUserValuefederated_provider = 22
    kReferenceValueGroup = 12
    kReferenceValueapp = 13
    kReferenceValuename_space = 20
    kReferenceValuePathElementGroup = 14
    kReferenceValuePathElementtype = 15
    kReferenceValuePathElementid = 16
    kReferenceValuePathElementname = 17
    kReferenceValuedatabase_id = 23
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'int64Value', 2: 'booleanValue', 3: 'stringValue', 4: 'doubleValue', 5: 'PointValue', 6: 'x', 7: 'y', 8: 'UserValue', 9: 'email', 10: 'auth_domain', 11: 'nickname', 12: 'ReferenceValue', 13: 'app', 14: 'PathElement', 15: 'type', 16: 'id', 17: 'name', 18: 'gaiaid', 19: 'obfuscated_gaiaid', 20: 'name_space', 21: 'federated_identity', 22: 'federated_provider', 23: 'database_id'}, 23)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.STRING, 4: ProtocolBuffer.Encoder.DOUBLE, 5: ProtocolBuffer.Encoder.STARTGROUP, 6: ProtocolBuffer.Encoder.DOUBLE, 7: ProtocolBuffer.Encoder.DOUBLE, 8: ProtocolBuffer.Encoder.STARTGROUP, 9: ProtocolBuffer.Encoder.STRING, 10: ProtocolBuffer.Encoder.STRING, 11: ProtocolBuffer.Encoder.STRING, 12: ProtocolBuffer.Encoder.STARTGROUP, 13: ProtocolBuffer.Encoder.STRING, 14: ProtocolBuffer.Encoder.STARTGROUP, 15: ProtocolBuffer.Encoder.STRING, 16: ProtocolBuffer.Encoder.NUMERIC, 17: ProtocolBuffer.Encoder.STRING, 18: ProtocolBuffer.Encoder.NUMERIC, 19: ProtocolBuffer.Encoder.STRING, 20: ProtocolBuffer.Encoder.STRING, 21: ProtocolBuffer.Encoder.STRING, 22: ProtocolBuffer.Encoder.STRING, 23: ProtocolBuffer.Encoder.STRING}, 23, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.PropertyValue'