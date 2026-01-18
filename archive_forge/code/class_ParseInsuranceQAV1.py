from typing import Optional
import gzip
import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
class ParseInsuranceQAV1(ParseInsuranceQA):
    version = 'V1'
    label2answer_fname = 'answers.label.token_idx'

    @classmethod
    def write_data_files(cls, dpext, out_path, d_vocab, d_label_answer):
        data_fnames = [('train', 'question.train.token_idx.label'), ('valid', 'question.dev.label.token_idx.pool'), ('test', 'question.test1.label.token_idx.pool')]
        for dtype, data_fname in data_fnames:
            data_path = os.path.join(dpext, data_fname)
            cls.create_fb_format(out_path, dtype, data_path, d_vocab, d_label_answer)

    @classmethod
    def create_fb_format(cls, out_path, dtype, inpath, d_vocab, d_label_answer):
        print('building fbformat:' + dtype)
        fout = open(os.path.join(out_path, dtype + '.txt'), 'w')
        lines = open(inpath).readlines()
        for line in lines:
            fields = line.rstrip('\n').split('\t')
            if dtype == 'train':
                assert len(fields) == 2, 'data file (%s) corrupted.' % inpath
                s_q_wids, s_good_aids = fields
                q = cls.wids2sent(s_q_wids.split(), d_vocab)
                good_ans = [d_label_answer[aid_] for aid_ in s_good_aids.split()]
                s = '1 ' + q + '\t' + '|'.join(good_ans)
                fout.write(s + '\n')
            else:
                assert len(fields) == 3, 'data file (%s) corrupted.' % inpath
                s_good_aids, s_q_wids, s_bad_aids = fields
                q = cls.wids2sent(s_q_wids.split(), d_vocab)
                good_ans = [d_label_answer[aid_] for aid_ in s_good_aids.split()]
                bad_ans = [d_label_answer[aid_] for aid_ in s_bad_aids.split()]
                s = '1 ' + q + '\t' + '|'.join(good_ans) + '\t\t' + '|'.join(good_ans + bad_ans)
                fout.write(s + '\n')
        fout.close()