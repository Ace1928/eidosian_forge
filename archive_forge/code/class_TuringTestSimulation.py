import random
class TuringTestSimulation:

    def __init__(self):
        self.roles = ['interrogator', 'human', 'machine']
        self.current_role = None
        self.questions = ['What is your favorite color?', 'Describe your last vacation.', 'How do you solve complex problems?', 'What does happiness mean to you?']

    def select_role(self):
        print('Select a role to play:')
        for index, role in enumerate(self.roles):
            print(f'{index + 1}. {role}')
        choice = int(input('Enter your choice (1-3): '))
        self.current_role = self.roles[choice - 1]
        print(f'You are now the {self.current_role}.')

    def generate_question(self):
        return random.choice(self.questions)

    def generate_response(self, question):
        if self.current_role == 'human':
            return input(f'Answer the question: {question}\n')
        elif self.current_role == 'machine':
            return 'I enjoy all colors equally as a machine without preference.'

    def run_session(self):
        if self.current_role == 'interrogator':
            other_role = random.choice(['human', 'machine'])
            print(f"You are questioning a {other_role}. Try to determine if it's human or machine.")
            for _ in range(4):
                question = self.generate_question()
                print(f'Question: {question}')
                response = self.generate_response(question)
                print(f'Response: {response}')
        else:
            for _ in range(4):
                question = self.generate_question()
                response = self.generate_response(question)
                print(f'Question: {question}\nYour Response: {response}')