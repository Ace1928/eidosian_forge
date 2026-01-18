import py7zr

# Files to include in the archive
files_to_archive = ["file1.txt", "file2.txt", "file3.txt"]

# Name of the 7z archive
archive_name = "example.7z"

# Create the 7z archive
with py7zr.SevenZipFile(archive_name, "w") as archive:
    for file in files_to_archive:
        archive.write(file)

print("7z archive created successfully.")

# Extract files from the 7z archive
with py7zr.SevenZipFile(archive_name, "r") as archive:
    archive.extractall()

print("Files extracted successfully.")
