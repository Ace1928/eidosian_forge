from io import BytesIO
from ... import tests
from .. import pack
class TestMakeReadvReader(tests.TestCaseWithTransport):

    def test_read_skipping_records(self):
        pack_data = BytesIO()
        writer = pack.ContainerWriter(pack_data.write)
        writer.begin()
        memos = []
        memos.append(writer.add_bytes_record([b'abc'], 3, names=[]))
        memos.append(writer.add_bytes_record([b'def'], 3, names=[(b'name1',)]))
        memos.append(writer.add_bytes_record([b'ghi'], 3, names=[(b'name2',)]))
        memos.append(writer.add_bytes_record([b'jkl'], 3, names=[]))
        writer.end()
        transport = self.get_transport()
        transport.put_bytes('mypack', pack_data.getvalue())
        requested_records = [memos[0], memos[2]]
        reader = pack.make_readv_reader(transport, 'mypack', requested_records)
        result = []
        for names, reader_func in reader.iter_records():
            result.append((names, reader_func(None)))
        self.assertEqual([([], b'abc'), ([(b'name2',)], b'ghi')], result)