from parlai.core.teachers import DialogTeacher
from .build import build
import os
class QAZRETeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        self.dt = opt['datatype'].split(':')[0]
        self.id = 'qazre'
        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'QA-ZRE', 'relation_splits')
        super().__init__(opt, shared)

    def setup_data(self, input_path):
        print('loading: ' + input_path)
        new_episode = True

        def extract_qa(qa_data):
            line_data = qa_data.split('\t')
            question_type, anon_question, deanon, context = line_data[:4]
            answer = line_data[4:]
            if answer == []:
                answer = ['No answer']
            return (context + '\n' + anon_question.replace('XXX', deanon), answer)
        for fname in os.listdir(input_path):
            if fname.startswith(self.dt):
                with open(os.path.join(input_path, fname)) as file:
                    file_data = file.read().split('\n')[:-1]
                for line in file_data:
                    question, answer = extract_qa(line)
                    yield ((question, answer, None, None), new_episode)