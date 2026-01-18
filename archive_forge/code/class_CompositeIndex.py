from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class CompositeIndex(ProtocolBuffer.ProtocolMessage):
    WRITE_ONLY = 1
    READ_WRITE = 2
    DELETED = 3
    ERROR = 4
    _State_NAMES = {1: 'WRITE_ONLY', 2: 'READ_WRITE', 3: 'DELETED', 4: 'ERROR'}

    def State_Name(cls, x):
        return cls._State_NAMES.get(x, '')
    State_Name = classmethod(State_Name)
    PENDING = 1
    ACTIVE = 2
    COMPLETED = 3
    _WorkflowState_NAMES = {1: 'PENDING', 2: 'ACTIVE', 3: 'COMPLETED'}

    def WorkflowState_Name(cls, x):
        return cls._WorkflowState_NAMES.get(x, '')
    WorkflowState_Name = classmethod(WorkflowState_Name)
    has_app_id_ = 0
    app_id_ = ''
    has_database_id_ = 0
    database_id_ = ''
    has_id_ = 0
    id_ = 0
    has_definition_ = 0
    has_state_ = 0
    state_ = 0
    has_workflow_state_ = 0
    workflow_state_ = 0
    has_error_message_ = 0
    error_message_ = ''
    has_only_use_if_required_ = 0
    only_use_if_required_ = 0
    has_disabled_index_ = 0
    disabled_index_ = 0
    has_deprecated_write_division_family_ = 0
    deprecated_write_division_family_ = ''

    def __init__(self, contents=None):
        self.definition_ = Index()
        self.deprecated_read_division_family_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def app_id(self):
        return self.app_id_

    def set_app_id(self, x):
        self.has_app_id_ = 1
        self.app_id_ = x

    def clear_app_id(self):
        if self.has_app_id_:
            self.has_app_id_ = 0
            self.app_id_ = ''

    def has_app_id(self):
        return self.has_app_id_

    def database_id(self):
        return self.database_id_

    def set_database_id(self, x):
        self.has_database_id_ = 1
        self.database_id_ = x

    def clear_database_id(self):
        if self.has_database_id_:
            self.has_database_id_ = 0
            self.database_id_ = ''

    def has_database_id(self):
        return self.has_database_id_

    def id(self):
        return self.id_

    def set_id(self, x):
        self.has_id_ = 1
        self.id_ = x

    def clear_id(self):
        if self.has_id_:
            self.has_id_ = 0
            self.id_ = 0

    def has_id(self):
        return self.has_id_

    def definition(self):
        return self.definition_

    def mutable_definition(self):
        self.has_definition_ = 1
        return self.definition_

    def clear_definition(self):
        self.has_definition_ = 0
        self.definition_.Clear()

    def has_definition(self):
        return self.has_definition_

    def state(self):
        return self.state_

    def set_state(self, x):
        self.has_state_ = 1
        self.state_ = x

    def clear_state(self):
        if self.has_state_:
            self.has_state_ = 0
            self.state_ = 0

    def has_state(self):
        return self.has_state_

    def workflow_state(self):
        return self.workflow_state_

    def set_workflow_state(self, x):
        self.has_workflow_state_ = 1
        self.workflow_state_ = x

    def clear_workflow_state(self):
        if self.has_workflow_state_:
            self.has_workflow_state_ = 0
            self.workflow_state_ = 0

    def has_workflow_state(self):
        return self.has_workflow_state_

    def error_message(self):
        return self.error_message_

    def set_error_message(self, x):
        self.has_error_message_ = 1
        self.error_message_ = x

    def clear_error_message(self):
        if self.has_error_message_:
            self.has_error_message_ = 0
            self.error_message_ = ''

    def has_error_message(self):
        return self.has_error_message_

    def only_use_if_required(self):
        return self.only_use_if_required_

    def set_only_use_if_required(self, x):
        self.has_only_use_if_required_ = 1
        self.only_use_if_required_ = x

    def clear_only_use_if_required(self):
        if self.has_only_use_if_required_:
            self.has_only_use_if_required_ = 0
            self.only_use_if_required_ = 0

    def has_only_use_if_required(self):
        return self.has_only_use_if_required_

    def disabled_index(self):
        return self.disabled_index_

    def set_disabled_index(self, x):
        self.has_disabled_index_ = 1
        self.disabled_index_ = x

    def clear_disabled_index(self):
        if self.has_disabled_index_:
            self.has_disabled_index_ = 0
            self.disabled_index_ = 0

    def has_disabled_index(self):
        return self.has_disabled_index_

    def deprecated_read_division_family_size(self):
        return len(self.deprecated_read_division_family_)

    def deprecated_read_division_family_list(self):
        return self.deprecated_read_division_family_

    def deprecated_read_division_family(self, i):
        return self.deprecated_read_division_family_[i]

    def set_deprecated_read_division_family(self, i, x):
        self.deprecated_read_division_family_[i] = x

    def add_deprecated_read_division_family(self, x):
        self.deprecated_read_division_family_.append(x)

    def clear_deprecated_read_division_family(self):
        self.deprecated_read_division_family_ = []

    def deprecated_write_division_family(self):
        return self.deprecated_write_division_family_

    def set_deprecated_write_division_family(self, x):
        self.has_deprecated_write_division_family_ = 1
        self.deprecated_write_division_family_ = x

    def clear_deprecated_write_division_family(self):
        if self.has_deprecated_write_division_family_:
            self.has_deprecated_write_division_family_ = 0
            self.deprecated_write_division_family_ = ''

    def has_deprecated_write_division_family(self):
        return self.has_deprecated_write_division_family_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_app_id():
            self.set_app_id(x.app_id())
        if x.has_database_id():
            self.set_database_id(x.database_id())
        if x.has_id():
            self.set_id(x.id())
        if x.has_definition():
            self.mutable_definition().MergeFrom(x.definition())
        if x.has_state():
            self.set_state(x.state())
        if x.has_workflow_state():
            self.set_workflow_state(x.workflow_state())
        if x.has_error_message():
            self.set_error_message(x.error_message())
        if x.has_only_use_if_required():
            self.set_only_use_if_required(x.only_use_if_required())
        if x.has_disabled_index():
            self.set_disabled_index(x.disabled_index())
        for i in range(x.deprecated_read_division_family_size()):
            self.add_deprecated_read_division_family(x.deprecated_read_division_family(i))
        if x.has_deprecated_write_division_family():
            self.set_deprecated_write_division_family(x.deprecated_write_division_family())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_app_id_ != x.has_app_id_:
            return 0
        if self.has_app_id_ and self.app_id_ != x.app_id_:
            return 0
        if self.has_database_id_ != x.has_database_id_:
            return 0
        if self.has_database_id_ and self.database_id_ != x.database_id_:
            return 0
        if self.has_id_ != x.has_id_:
            return 0
        if self.has_id_ and self.id_ != x.id_:
            return 0
        if self.has_definition_ != x.has_definition_:
            return 0
        if self.has_definition_ and self.definition_ != x.definition_:
            return 0
        if self.has_state_ != x.has_state_:
            return 0
        if self.has_state_ and self.state_ != x.state_:
            return 0
        if self.has_workflow_state_ != x.has_workflow_state_:
            return 0
        if self.has_workflow_state_ and self.workflow_state_ != x.workflow_state_:
            return 0
        if self.has_error_message_ != x.has_error_message_:
            return 0
        if self.has_error_message_ and self.error_message_ != x.error_message_:
            return 0
        if self.has_only_use_if_required_ != x.has_only_use_if_required_:
            return 0
        if self.has_only_use_if_required_ and self.only_use_if_required_ != x.only_use_if_required_:
            return 0
        if self.has_disabled_index_ != x.has_disabled_index_:
            return 0
        if self.has_disabled_index_ and self.disabled_index_ != x.disabled_index_:
            return 0
        if len(self.deprecated_read_division_family_) != len(x.deprecated_read_division_family_):
            return 0
        for e1, e2 in zip(self.deprecated_read_division_family_, x.deprecated_read_division_family_):
            if e1 != e2:
                return 0
        if self.has_deprecated_write_division_family_ != x.has_deprecated_write_division_family_:
            return 0
        if self.has_deprecated_write_division_family_ and self.deprecated_write_division_family_ != x.deprecated_write_division_family_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_app_id_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: app_id not set.')
        if not self.has_id_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: id not set.')
        if not self.has_definition_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: definition not set.')
        elif not self.definition_.IsInitialized(debug_strs):
            initialized = 0
        if not self.has_state_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: state not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.app_id_))
        if self.has_database_id_:
            n += 1 + self.lengthString(len(self.database_id_))
        n += self.lengthVarInt64(self.id_)
        n += self.lengthString(self.definition_.ByteSize())
        n += self.lengthVarInt64(self.state_)
        if self.has_workflow_state_:
            n += 1 + self.lengthVarInt64(self.workflow_state_)
        if self.has_error_message_:
            n += 1 + self.lengthString(len(self.error_message_))
        if self.has_only_use_if_required_:
            n += 2
        if self.has_disabled_index_:
            n += 2
        n += 1 * len(self.deprecated_read_division_family_)
        for i in range(len(self.deprecated_read_division_family_)):
            n += self.lengthString(len(self.deprecated_read_division_family_[i]))
        if self.has_deprecated_write_division_family_:
            n += 1 + self.lengthString(len(self.deprecated_write_division_family_))
        return n + 4

    def ByteSizePartial(self):
        n = 0
        if self.has_app_id_:
            n += 1
            n += self.lengthString(len(self.app_id_))
        if self.has_database_id_:
            n += 1 + self.lengthString(len(self.database_id_))
        if self.has_id_:
            n += 1
            n += self.lengthVarInt64(self.id_)
        if self.has_definition_:
            n += 1
            n += self.lengthString(self.definition_.ByteSizePartial())
        if self.has_state_:
            n += 1
            n += self.lengthVarInt64(self.state_)
        if self.has_workflow_state_:
            n += 1 + self.lengthVarInt64(self.workflow_state_)
        if self.has_error_message_:
            n += 1 + self.lengthString(len(self.error_message_))
        if self.has_only_use_if_required_:
            n += 2
        if self.has_disabled_index_:
            n += 2
        n += 1 * len(self.deprecated_read_division_family_)
        for i in range(len(self.deprecated_read_division_family_)):
            n += self.lengthString(len(self.deprecated_read_division_family_[i]))
        if self.has_deprecated_write_division_family_:
            n += 1 + self.lengthString(len(self.deprecated_write_division_family_))
        return n

    def Clear(self):
        self.clear_app_id()
        self.clear_database_id()
        self.clear_id()
        self.clear_definition()
        self.clear_state()
        self.clear_workflow_state()
        self.clear_error_message()
        self.clear_only_use_if_required()
        self.clear_disabled_index()
        self.clear_deprecated_read_division_family()
        self.clear_deprecated_write_division_family()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.app_id_)
        out.putVarInt32(16)
        out.putVarInt64(self.id_)
        out.putVarInt32(26)
        out.putVarInt32(self.definition_.ByteSize())
        self.definition_.OutputUnchecked(out)
        out.putVarInt32(32)
        out.putVarInt32(self.state_)
        if self.has_only_use_if_required_:
            out.putVarInt32(48)
            out.putBoolean(self.only_use_if_required_)
        for i in range(len(self.deprecated_read_division_family_)):
            out.putVarInt32(58)
            out.putPrefixedString(self.deprecated_read_division_family_[i])
        if self.has_deprecated_write_division_family_:
            out.putVarInt32(66)
            out.putPrefixedString(self.deprecated_write_division_family_)
        if self.has_disabled_index_:
            out.putVarInt32(72)
            out.putBoolean(self.disabled_index_)
        if self.has_workflow_state_:
            out.putVarInt32(80)
            out.putVarInt32(self.workflow_state_)
        if self.has_error_message_:
            out.putVarInt32(90)
            out.putPrefixedString(self.error_message_)
        if self.has_database_id_:
            out.putVarInt32(98)
            out.putPrefixedString(self.database_id_)

    def OutputPartial(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        if self.has_id_:
            out.putVarInt32(16)
            out.putVarInt64(self.id_)
        if self.has_definition_:
            out.putVarInt32(26)
            out.putVarInt32(self.definition_.ByteSizePartial())
            self.definition_.OutputPartial(out)
        if self.has_state_:
            out.putVarInt32(32)
            out.putVarInt32(self.state_)
        if self.has_only_use_if_required_:
            out.putVarInt32(48)
            out.putBoolean(self.only_use_if_required_)
        for i in range(len(self.deprecated_read_division_family_)):
            out.putVarInt32(58)
            out.putPrefixedString(self.deprecated_read_division_family_[i])
        if self.has_deprecated_write_division_family_:
            out.putVarInt32(66)
            out.putPrefixedString(self.deprecated_write_division_family_)
        if self.has_disabled_index_:
            out.putVarInt32(72)
            out.putBoolean(self.disabled_index_)
        if self.has_workflow_state_:
            out.putVarInt32(80)
            out.putVarInt32(self.workflow_state_)
        if self.has_error_message_:
            out.putVarInt32(90)
            out.putPrefixedString(self.error_message_)
        if self.has_database_id_:
            out.putVarInt32(98)
            out.putPrefixedString(self.database_id_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_app_id(d.getPrefixedString())
                continue
            if tt == 16:
                self.set_id(d.getVarInt64())
                continue
            if tt == 26:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_definition().TryMerge(tmp)
                continue
            if tt == 32:
                self.set_state(d.getVarInt32())
                continue
            if tt == 48:
                self.set_only_use_if_required(d.getBoolean())
                continue
            if tt == 58:
                self.add_deprecated_read_division_family(d.getPrefixedString())
                continue
            if tt == 66:
                self.set_deprecated_write_division_family(d.getPrefixedString())
                continue
            if tt == 72:
                self.set_disabled_index(d.getBoolean())
                continue
            if tt == 80:
                self.set_workflow_state(d.getVarInt32())
                continue
            if tt == 90:
                self.set_error_message(d.getPrefixedString())
                continue
            if tt == 98:
                self.set_database_id(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_app_id_:
            res += prefix + 'app_id: %s\n' % self.DebugFormatString(self.app_id_)
        if self.has_database_id_:
            res += prefix + 'database_id: %s\n' % self.DebugFormatString(self.database_id_)
        if self.has_id_:
            res += prefix + 'id: %s\n' % self.DebugFormatInt64(self.id_)
        if self.has_definition_:
            res += prefix + 'definition <\n'
            res += self.definition_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_state_:
            res += prefix + 'state: %s\n' % self.DebugFormatInt32(self.state_)
        if self.has_workflow_state_:
            res += prefix + 'workflow_state: %s\n' % self.DebugFormatInt32(self.workflow_state_)
        if self.has_error_message_:
            res += prefix + 'error_message: %s\n' % self.DebugFormatString(self.error_message_)
        if self.has_only_use_if_required_:
            res += prefix + 'only_use_if_required: %s\n' % self.DebugFormatBool(self.only_use_if_required_)
        if self.has_disabled_index_:
            res += prefix + 'disabled_index: %s\n' % self.DebugFormatBool(self.disabled_index_)
        cnt = 0
        for e in self.deprecated_read_division_family_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'deprecated_read_division_family%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        if self.has_deprecated_write_division_family_:
            res += prefix + 'deprecated_write_division_family: %s\n' % self.DebugFormatString(self.deprecated_write_division_family_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kapp_id = 1
    kdatabase_id = 12
    kid = 2
    kdefinition = 3
    kstate = 4
    kworkflow_state = 10
    kerror_message = 11
    konly_use_if_required = 6
    kdisabled_index = 9
    kdeprecated_read_division_family = 7
    kdeprecated_write_division_family = 8
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'app_id', 2: 'id', 3: 'definition', 4: 'state', 6: 'only_use_if_required', 7: 'deprecated_read_division_family', 8: 'deprecated_write_division_family', 9: 'disabled_index', 10: 'workflow_state', 11: 'error_message', 12: 'database_id'}, 12)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.STRING, 4: ProtocolBuffer.Encoder.NUMERIC, 6: ProtocolBuffer.Encoder.NUMERIC, 7: ProtocolBuffer.Encoder.STRING, 8: ProtocolBuffer.Encoder.STRING, 9: ProtocolBuffer.Encoder.NUMERIC, 10: ProtocolBuffer.Encoder.NUMERIC, 11: ProtocolBuffer.Encoder.STRING, 12: ProtocolBuffer.Encoder.STRING}, 12, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.CompositeIndex'