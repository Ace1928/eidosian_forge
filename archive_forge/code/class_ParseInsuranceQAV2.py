from typing import Optional
import gzip
import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
class ParseInsuranceQAV2(ParseInsuranceQA):
    version = 'V2'
    label2answer_fname = 'InsuranceQA.label2answer.token.encoded.gz'

    @classmethod
    def write_data_files(cls, dpext, out_path, d_vocab, d_label_answer):
        data_fnames_tmpl = [('train.%s', 'InsuranceQA.question.anslabel.token.%s.pool.solr.train.encoded.gz'), ('valid.%s', 'InsuranceQA.question.anslabel.token.%s.pool.solr.valid.encoded.gz'), ('test.%s', 'InsuranceQA.question.anslabel.token.%s.pool.solr.test.encoded.gz')]
        for n_cands in [100, 500, 1000, 1500]:
            for dtype_tmp, data_fname_tmp in data_fnames_tmpl:
                dtype = dtype_tmp % n_cands
                data_fname = data_fname_tmp % n_cands
                data_path = os.path.join(dpext, data_fname)
                cls.create_fb_format(out_path, dtype, data_path, d_vocab, d_label_answer)

    @classmethod
    def create_fb_format(cls, out_path, dtype, inpath, d_vocab, d_label_answer):
        print('building fbformat:' + dtype)
        fout = open(os.path.join(out_path, dtype + '.txt'), 'w')
        lines = cls.readlines(inpath)
        for line in lines:
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 4:
                raise ValueError('data file (%s) corrupted. Line (%s)' % (repr(line), inpath))
            else:
                _, s_q_wids, s_good_aids, s_bad_aids = fields
                q = cls.wids2sent(s_q_wids.split(), d_vocab)
                good_ans = [d_label_answer[aid_] for aid_ in s_good_aids.split()]
                bad_ans = [d_label_answer[aid_] for aid_ in s_bad_aids.split()]
                s = '1 ' + q + '\t' + '|'.join(good_ans) + '\t\t' + '|'.join(good_ans + bad_ans)
                fout.write(s + '\n')
        fout.close()